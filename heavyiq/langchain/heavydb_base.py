# Contain base classes of HeavyDB class
import asyncio
import functools
import logging
import multiprocessing
import re
import threading
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from multiprocessing.managers import SyncManager
from typing import Any, Callable, Optional, ParamSpec, TypedDict, TypeVar

import anyio
from heavyai import Connection, connect
from heavydb._parsers import ColumnDetails

from heavyiq.config import get_config
from heavyiq.langchain.exceptions import HeavyDBTimeoutError
from heavyiq.utils import LRUCache, calc_query_stats, is_destructive_sql, rate_sql_complexity, strip_sql_comments

T = TypeVar("T", bound="HeavyDBCacheBase")
C = TypeVar("C", bound="HeavyDBCreateBase")
P = ParamSpec("P")
R = TypeVar("R")

config = get_config()


class StringLiteralOp(TypedDict):
    """
    A type representation for string literal operations.
    """

    operator: str
    literal: str
    database: str
    table: str
    column: str


class HeavyDBCacheBase:
    """
    Contain attributes and method which are relevant to caches.
    Here we use caches (single multiprocessing.manager instance as base) for storing top_k, timestamp, etc.
    """

    _manager: Optional[SyncManager] = None
    _top_k_cache: Optional[LRUCache[str, str]] = None
    _sample_rows_cache: Optional[LRUCache[str, str]] = None
    _table_schema_cache: Optional[LRUCache[str, str]] = None
    _table_text_columns_count_cache: Optional[LRUCache[str, int]] = None
    _table_total_row_count_cache: Optional[LRUCache[str, int]] = None
    _timestamp_cache: Optional[LRUCache[str, str]] = None

    @classmethod
    def get_manager(cls: type[T]) -> SyncManager:
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
        return cls._manager

    @classmethod
    def initialize(cls: type[T]) -> None:
        """
        Initializes all the inter-process caches.
        """
        cls.get_top_k_cache()
        cls.get_sample_rows_cache()
        cls.get_table_schema_cache()
        cls.get_table_text_columns_count_cache()
        cls.get_table_total_row_count_cache()
        cls.get_timestamp_cache()

    @classmethod
    def get_top_k_cache(cls: type[T]) -> LRUCache[str, str]:
        if cls._top_k_cache is None:
            cls._top_k_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._top_k_cache

    @classmethod
    def get_sample_rows_cache(cls: type[T]) -> LRUCache[str, str]:
        if cls._sample_rows_cache is None:
            cls._sample_rows_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._sample_rows_cache

    @classmethod
    def get_table_schema_cache(cls: type[T]) -> LRUCache[str, str]:
        if cls._table_schema_cache is None:
            cls._table_schema_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._table_schema_cache

    @classmethod
    def get_table_text_columns_count_cache(cls: type[T]) -> LRUCache[str, int]:
        if cls._table_text_columns_count_cache is None:
            cls._table_text_columns_count_cache = LRUCache[str, int](manager=cls.get_manager())
        return cls._table_text_columns_count_cache

    @classmethod
    def get_table_total_row_count_cache(cls: type[T]) -> LRUCache[str, int]:
        if cls._table_total_row_count_cache is None:
            cls._table_total_row_count_cache = LRUCache[str, int](manager=cls.get_manager())
        return cls._table_total_row_count_cache

    @classmethod
    def get_timestamp_cache(cls: type[T]) -> LRUCache[str, str]:
        if cls._timestamp_cache is None:
            cls._timestamp_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._timestamp_cache

    @property
    def top_k_cache(self) -> LRUCache[str, str]:
        return self.get_top_k_cache()

    @property
    def sample_rows_cache(self) -> LRUCache[str, str]:
        return self.get_sample_rows_cache()

    @property
    def table_schema_cache(self) -> LRUCache[str, str]:
        return self.get_table_schema_cache()

    @property
    def table_text_columns_count_cache(self) -> LRUCache[str, int]:
        return self.get_table_text_columns_count_cache()

    @property
    def table_total_row_count_cache(self) -> LRUCache[str, int]:
        return self.get_table_total_row_count_cache()

    @property
    def timestamp_cache(self) -> LRUCache[str, str]:
        return self.get_timestamp_cache()


class PersistantConnection(Connection):
    """
    Helps to establish heavydb connection which persists for atleast an hour.

    Default heavyai.Connection gets closed once the corresponding __del__ method being called (ie. upon garbage collection).
    So we can't re-use the same session_id on `/query`, `/question` endpoints.

    This class helps to overcome the above issue, and the created session gets auto expire after certain limit
    (Default session-timeout-value set on server side is 60 mins).
    """

    def close(self):
        """Don't disconnect from the database. Let the session expire automatically."""
        self._closed = 1
        self._rbc = None


class DBConnectionPool:
    """
    Helps to create a pool of HeavyDB connection objects.
    """

    def __init__(self, create_conn: Callable, maxsize: int = 20):
        self._create_conn = create_conn
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def acquire(self) -> Connection:
        try:
            # Try to get a connection without waiting
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Pool is empty: create a new connection
            # always pass coroutine as create func
            conn = await self._create_conn()
        return conn

    async def release(self, conn: Connection):
        try:
            # Try to put it back in the pool
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            # Pool full: close extra connections
            conn.close()

    async def close(self):
        while not self._pool.empty():
            conn = await self._pool.get()
            conn.close()

    @asynccontextmanager
    async def connection(self):
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)


class HeavyDBCreateBase:
    """
    Contain methods for creating HeavyDB cls instances.
    """

    @classmethod
    def _connect_with_timeout(cls: type[C], connect_func: Callable[[], Connection], timeout: float = 10) -> Connection:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(connect_func)
            try:
                return future.result(timeout)
            except TimeoutError:
                raise Exception(f"HeavyDB failed to connect after {timeout} seconds")

    @classmethod
    async def _aconnect_with_timeout(cls: type[C], coroutine: Awaitable[Connection], timeout: float = 10) -> Connection:
        """
        Runs a coroutine inside a CancellableScope.
        """
        connection = None
        with anyio.move_on_after(timeout) as scope:
            connection = await coroutine
        if scope.cancel_called:
            raise Exception(f"HeavyDB failed to connect after {timeout} seconds")
        return connection

    @classmethod
    async def create_connection(
        cls: type[C], persistant: bool = False, timeout: int = 10, **kwargs
    ) -> Connection | PersistantConnection:
        """
        Helps to establish HeavyDB connection.
        """
        connection = None
        with anyio.move_on_after(timeout) as _:
            connect_func = PersistantConnection if persistant else connect
            func = functools.partial(connect_func, **kwargs)
            connection = await anyio.to_thread.run_sync(func, cancellable=True)
        if not connection:
            raise Exception(f"HeavyDB failed to connect after {timeout} seconds")
        return connection

    @classmethod
    async def call_client(
        cls: type[C], client_method: Callable[P, R], *args: P.args, timeout: int = 10, **kwargs: P.kwargs
    ) -> R:
        """
        Helps to call synchronous HeavyDB client methods from async context with timeout.
        """
        with anyio.move_on_after(timeout) as scope:
            func = functools.partial(client_method, *args, **kwargs)
            response = await anyio.to_thread.run_sync(func, cancellable=True)
        if scope.cancel_called:
            raise HeavyDBTimeoutError(
                f'HeavyDB client method "{client_method.__str__}" failed to generate response within {timeout} seconds.'
            )
        return response

    @classmethod
    async def create_connection_from_env_async(
        cls: type[C], db_name: Optional[str] = None, timeout: int = 10
    ) -> Connection:
        """
        Creates HeavyDB connection from env async.
        """
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        conn = await cls.create_connection(
            persistant=False,
            timeout=timeout,
            user=config.heavydb_username,
            password=config.heavydb_password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            dbname=db_name or config.heavydb_dbname,
            protocol=config.heavydb_protocol,
        )
        return conn

    @classmethod
    async def create_connection_from_session_async(cls: type[C], session_id: str, timeout: int = 10) -> Connection:
        """
        Established HeavyDB connection from sesssion id.
        """
        conn = await cls.create_connection(
            persistant=False,
            timeout=timeout,
            sessionid=session_id,
            host=config.heavydb_host,
            port=config.heavydb_port,
            protocol=config.heavydb_protocol,
        )
        return conn

    @classmethod
    async def from_env_async(cls: type[C], db_name: Optional[str] = None, **kwargs: Any) -> C:
        """Create a database connection from environment variables."""
        timeout = kwargs.pop("timeout", 10)
        conn = await cls.create_connection_from_env_async(db_name=db_name, timeout=timeout)
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    async def from_session_async(cls: type[C], session_id: str, **kwargs: Any) -> C:
        """Create a database connection from session."""
        timeout = kwargs.pop("timeout", 10)
        conn = await cls.create_connection_from_session_async(session_id=session_id, timeout=timeout)
        # helps to grab the dbname because conn._dbname woudn't be available if the connection was created using session_id
        session_info = await cls.call_client(conn._client.get_session_info, conn._session)
        dbname = session_info.database
        return cls(conn, dbname=dbname, **kwargs)  # type: ignore

    @classmethod
    def from_uri(cls: type[C], database_uri: str, **kwargs: Any) -> C:
        """Create a database connection from a database URI."""
        conn = cls._connect_with_timeout(lambda: connect(database_uri), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    def from_env(cls: type[C], db_name: Optional[str] = None, **kwargs: Any) -> C:
        """Create a database connection from environment variables."""
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        def connect_func() -> Connection:
            return connect(
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=db_name or config.heavydb_dbname,
                protocol=config.heavydb_protocol,
            )

        conn = cls._connect_with_timeout(connect_func, kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    def from_session(cls: type[C], session_id: str, **kwargs: Any) -> C:
        """Create a database connection from a session id."""

        def connect_func() -> Connection:
            return connect(
                sessionid=session_id,
                host=config.heavydb_host,
                port=config.heavydb_port,
                protocol=config.heavydb_protocol,
            )

        conn = cls._connect_with_timeout(connect_func, kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    def from_creds(cls: type[C], username: str, password: str, dbname: Optional[str] = None, **kwargs: Any) -> C:
        """Create a database connection from username and password."""
        if not config.heavydb_dbname and not dbname:
            raise ValueError(
                "Please set the config variable heavydb_dbname or provide a dbname as an argument to HeavyDB.from_creds"
            )
        conn: Connection = connect(
            user=username,
            password=password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            protocol=config.heavydb_protocol,
            dbname=dbname or config.heavydb_dbname,
        )
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    def create_session_id(cls: type[C], db_name: str | None = None) -> str:
        """
        Create a new db session and return it's id.
        """
        if not config.heavydb_username or not config.heavydb_password:
            raise ValueError("Please set the config variables heavydb_username and heavydb_password")

        conn: PersistantConnection = PersistantConnection(
            user=config.heavydb_username,
            password=config.heavydb_password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            protocol=config.heavydb_protocol,
            dbname=db_name or config.heavydb_dbname,
        )
        return conn._session

    @classmethod
    async def create_with_persistant_connection_async(cls: type[C], db_name: Optional[str] = None, **kwargs: Any) -> C:
        """
        Creates a HeavyDB instance from env with a persistant db connection.
        """
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        timeout = kwargs.pop("timeout", 10)
        conn = await cls.create_connection(
            persistant=True,
            timeout=timeout,
            user=config.heavydb_username,
            password=config.heavydb_password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            dbname=db_name or config.heavydb_dbname,
            protocol=config.heavydb_protocol,
        )
        return cls(conn, **kwargs)  # type: ignore

    @classmethod
    async def create_session_id_async(cls: type[C], db_name: Optional[str] = None, **kwargs: Any) -> str:
        """Creates a persistant database connection from environment variables and returns it's session id."""
        conn = await cls.create_with_persistant_connection_async(db_name=db_name, **kwargs)
        return conn._session  # type: ignore


class HeavyDBState:
    """
    Base class where the HeavyDB initialisation should happen.
    """

    __slots__ = (
        "_conn",
        "lock",
        "_dbname",
        "_all_tables",
        "_include_tables",
        "_ignore_tables",
        "_sample_rows_in_table_info",
        "_custom_table_info",
    )

    def __init__(
        self,
        conn: Connection,
        dbname: Optional[str] = None,
        ignore_tables: Optional[list[str]] = None,
        include_tables: Optional[list[str]] = None,
        sample_rows_in_table_info: int = 2,
        custom_table_info: Optional[dict[str, str]] = None,
    ) -> None:
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._conn = conn
        self.lock = threading.Lock()
        self._dbname = self._conn._dbname or dbname
        """Async lock ensures exactly one coroutine was allowed to access a shared resource (db) at a time."""

        self._all_tables = set(self._conn.get_tables())
        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(f"include_tables {missing_tables} not found in database")
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                print(f"WARNING: ignore_tables {missing_tables} not found in database")

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table]) for table in self._custom_table_info if table in intersection
            )

        # populate connection's _db_name attribute
        if not self._dbname:
            # connection established from session, so grab the session details
            self._dbname = self._conn._client.get_session_info(self._conn._session).database

    @functools.cached_property
    def logger(self) -> logging.Logger:
        from heavyiq.logging_utils import get_heavyiq_logger

        return get_heavyiq_logger()  # type: ignore


class HeavyDBOperationsSync(HeavyDBState, HeavyDBCacheBase):
    """
    Contain HeavyDB operations in sync methods.
    """

    def get_usable_table_names(self) -> set[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    def get_table_columns(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        self.logger.debug(f"Getting columns for table {table}")
        with self.lock:
            column_details = self._conn.get_column_details(table)
            self.logger.debug(f"Got columns for table {table}")
            return column_details

    def get_table_schema(self, table: str) -> str:
        self.logger.debug(f"Getting schema for table {table}")
        cache_key = f"{self._dbname}.{table}"
        cached_value = self.table_schema_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got schema for table {table} from cache")
            return cached_value
        """Get the schema of a table."""
        create_command = f'SHOW CREATE TABLE "{table}";'
        with self.lock:
            cursor = self._conn.execute(create_command)
        table_schema = cursor.fetchone()[0]  # type: ignore
        table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)  # type: ignore
        table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
        table_schema = re.sub(r"\n", "", table_schema)
        table_schema = re.sub(r"\( +", "(", table_schema)
        table_schema = re.sub(r", +", ", ", table_schema)
        if "WITH (" in table_schema:
            table_schema = table_schema[: table_schema.index("WITH (")]
        self.table_schema_cache.put(cache_key, table_schema)
        self.logger.debug(f"Got schema for table {table}")
        return table_schema

    def validate_query(self, query: str) -> list:
        """Validate a query."""
        # Remove block comments
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        if "::" in query:
            raise ValueError("Double colon cast syntax is not allowed. Use CAST() instead.")
        with self.lock:
            return self._conn._client.sql_validate(self._conn._session, query)

    def get_column_top_k(self, table: str, column: str, _k: int = 5) -> tuple[Optional[list[str]], bool]:
        config = get_config()
        cardinality_threshold = config.column_top_k_cardinality_threshold
        high_cardinality_sample = config.column_top_k_high_cardinality_sample

        column = f'"{column}"'
        # check to see if the column is low cardinality
        # fetch top (threshold + 1)
        # if there are < (threshold + 1) values, it's low cardinality and we can return all of them
        # if there are >= (threshold + 1) values, it's high cardinality and we need to sample the top high_cardinality_sample
        """Get the top k values for a column."""
        self.logger.debug(f"Getting top k values for column {column} in table {table}")
        top_k_statement = f'SELECT {column}, COUNT(*) as cnt FROM "{table}" WHERE {column} is not null GROUP BY {column} ORDER BY cnt DESC LIMIT {cardinality_threshold + 1};'
        with self.lock:
            cursor = self._conn.execute(top_k_statement)
        top_k_res: list[str] = [str(v[0]).replace("\n", " ") for v in cursor.fetchall()]
        is_high_cardinality = len(top_k_res) > cardinality_threshold
        if is_high_cardinality:
            # high-cardinality, return sample
            self.logger.debug(f"Column {column} is high-cardinality. Returning top {high_cardinality_sample} values")
            top_k_res = top_k_res[:high_cardinality_sample]
        else:
            self.logger.debug(f"Column {column} is low-cardinality.")
        self.logger.debug(f"Got top k values for column {column} in table {table}")
        if not any([v for v in top_k_res if v.startswith("MULTIPOLYGON")]):
            return top_k_res, is_high_cardinality
        return None, is_high_cardinality

    def get_sample_rows(self, table_name: str) -> str:
        self.logger.debug(f"Getting sample rows for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        cached_value = self.sample_rows_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got sample rows for table {table_name} from cache")
            return cached_value
        # build the select command
        command = f'SELECT * FROM "{table_name}" LIMIT {self._sample_rows_in_table_info}'

        # save the columns in string format
        columns_str = ",".join([col.name for col in self.get_table_columns(table_name)])

        # get the sample rows
        with self.lock:
            sample_rows = self._conn.execute(command)
        # shorten values in the sample rows
        sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

        # save the sample rows in string format
        sample_rows_str = "\n".join([",".join(row) for row in sample_rows])

        res = f"{self._sample_rows_in_table_info} rows from {table_name} table:\n{columns_str}\n{sample_rows_str}"

        self.sample_rows_cache.put(cache_key, res)
        self.logger.debug(f"Got sample rows for table {table_name}")
        return res

    def get_top_k(self, table_name: str) -> str:
        self.logger.debug(f"Getting top k values for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        cached_value = self.top_k_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got top k values for table {table_name} from cache")
            return cached_value
        text_columns = [
            c.name
            for c in self.get_table_columns(table_name)
            if c.type == "STR" and c.encoding == "DICT" and c.is_array is False
        ]
        low_cardinality_columns = []
        high_cardinality_columns = []
        for col in text_columns:
            top_k_res, is_high_cardinality = self.get_column_top_k(table_name, col)
            if top_k_res:
                if is_high_cardinality:
                    high_cardinality_columns.append((col, top_k_res))
                else:
                    low_cardinality_columns.append((col, top_k_res))
        top_k_strings = ""
        if low_cardinality_columns:
            top_k_strings += "Low cardinality columns and every possible value:\n"
            for col, top_k_res in low_cardinality_columns:
                top_k_strings += f"{col}: {', '.join(top_k_res)}\n"
        if high_cardinality_columns:
            top_k_strings += "High cardinality columns and most common values:\n"
            for col, top_k_res in high_cardinality_columns:
                top_k_strings += f"{col}: {', '.join(top_k_res)}\n"
        self.top_k_cache.put(cache_key, top_k_strings)
        self.logger.debug(f"Got top k values for table {table_name}")
        return top_k_strings

    def get_single_table_info(self, table_name: str, include_samples: bool = True, include_top_k: bool = True) -> str:
        """Get information about a single table."""
        all_table_names = self.get_usable_table_names()
        if table_name not in all_table_names:
            raise ValueError(f"table_name {table_name} not found in database")

        if self._custom_table_info and table_name in self._custom_table_info:
            return self._custom_table_info[table_name]

        parts = []

        parts.append(self.get_table_schema(table_name))

        if include_samples and self._sample_rows_in_table_info:
            parts.append(self.get_sample_rows(table_name))

        if include_top_k:
            parts.append(self.get_top_k(table_name))

        return "\n".join(parts)

    def get_table_info(self, table_names: Optional[list[str]] = None, **kwargs) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names  # type: ignore

        tables = []
        for table in all_table_names:
            tables.append(self.get_single_table_info(table, **kwargs))

        final_str = "\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all", to_str: bool = True) -> str | tuple | list:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        If to_str is True (by default), result should be converted to string.
        If fetch = one and to_str = False, then a tuple will be returned.
        If fetch = all and to_str = False, then a list of tuples will be returned.
        """
        command = strip_sql_comments(command)
        if is_destructive_sql(command):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(command)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")
        if to_str:
            return str(result)
        return result  # type: ignore

    def get_query_plan(self, query: str) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(f"EXPLAIN plan {query}")
        result: tuple[str] = cursor.fetchone()  # type: ignore
        return str(result[0])

    def get_calcite_query_plan(self, query: str, detailed: bool = False) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        sql_stmt = f"EXPLAIN CALCITE DETAILED {query}" if detailed else f"EXPLAIN CALCITE {query}"
        with self.lock:
            cursor = self._conn.execute(sql_stmt)
        query_plan: tuple[str] = cursor.fetchone()  # type: ignore
        return str(query_plan[0])

    def extract_column_mappings(self, detailed_query_plan: str) -> dict[str, tuple[str, str, str]]:
        self.logger.debug(f"Extracting column mappings from query plan: {detailed_query_plan}")
        result = {}
        # Use a regex pattern to capture (database, table, column) and literal values in LogicalFilter
        pattern = r"\[\$([0-9]+)->db:([\w]+),tableName:([\w]+),colName:([\w]+)\]"
        matches = re.findall(pattern, detailed_query_plan)
        for id, db, table, column in matches:
            col_id = (db, table, column)
            result[id] = col_id

        self.logger.debug("Finished extracting column mappings")
        return result

    def extract_string_literal_ops(self, detailed_query_plan: str) -> dict[str, tuple[str, str]]:
        self.logger.debug(f"Extracting string literal operations from query plan: {detailed_query_plan}")
        result = {}
        pattern1 = r"(NOT\()?(LIKE|PG_ILIKE|>=|<=|<>|=)\(\$(\d+),\s*'((?:''|[^'])+)'\)"
        pattern2 = r"(NOT\()?(LIKE|PG_ILIKE|>=|<=|<>|=)\('((?:''|[^'])+)',\s*\$(\d+)\)"

        matches1 = re.findall(pattern1, detailed_query_plan)
        matches2 = re.findall(pattern2, detailed_query_plan)
        for op, id, literal in matches1:
            result[id] = (op, literal)
        for op, literal, id in matches2:
            result[id] = (op, literal)

        self.logger.debug("Finished extracting string literal operations")
        return result

    def get_string_literal_ops(self, query: str) -> list[StringLiteralOp]:
        detailed_query_plan = self.get_calcite_query_plan(query, detailed=True)
        col_mapping = self.extract_column_mappings(detailed_query_plan)
        str_literal_ops = self.extract_string_literal_ops(detailed_query_plan)
        result: list[StringLiteralOp] = []
        for id, (op, literal) in str_literal_ops.items():
            db, table, column = col_mapping[id]
            result.append({"operator": op, "literal": literal, "database": db, "table": table, "column": column})
        return result

    def correct_string_literal(self, literal: StringLiteralOp, exact_match_threshold: float) -> StringLiteralOp:
        """
        Tries to correct a given string literal by searching for close matches in a database.

        Parameters
        ----------
        literal : StringLiteralOp
            A dictionary representing the string literal to be corrected.
            Expected keys are 'column', 'database', 'table', 'literal', and 'operator'.

        exact_match_threshold : float
            A threshold value between 0 and 1 that determines how strict the exact match ratio should be
            before deciding to change the literal or operator. If the exact match ratio is below this
            threshold, the operator might be changed to "ILIKE" or "NOT ILIKE".
        """
        self.logger.debug(f"Correcting string literal: {literal['column']} : {literal['literal']}")
        case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE '{literal['literal']}' GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
        with self.lock:
            cursor = self._conn.execute(case_match_query)
        case_match_rows = cursor.fetchall()

        num_case_match_rows = len(case_match_rows)
        exact_match_count = 0
        total_count = 0
        altered_literal = deepcopy(literal)

        for row in case_match_rows:
            if row[0] == literal["literal"]:
                exact_match_count += row[1]  # type: ignore
            total_count += row[1]  # type: ignore

        if total_count > 0 and literal["operator"] != "ILIKE":
            if exact_match_count == 0 and num_case_match_rows == 1:
                altered_literal["literal"] = str(case_match_rows[0][0])
                return literal
            elif exact_match_count / total_count < exact_match_threshold:
                if literal["operator"] == "<>":
                    altered_literal["operator"] = "NOT ILIKE"
                else:
                    altered_literal["operator"] = "ILIKE"
                return altered_literal
        elif total_count == 0:
            lower_literal = literal["literal"].lower()
            lower_literal_prefix = re.split(r"[ ,:]+", lower_literal)[0]
            using_lower_literal_prefix = False if lower_literal_prefix == lower_literal else True
            prefix_condition = (
                f"lower_attr ILIKE '{lower_literal_prefix}' OR lower_attr ILIKE '{lower_literal_prefix} %'"
                if using_lower_literal_prefix
                else f"lower_attr ILIKE '{lower_literal_prefix}%'"
            )

            prefix_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, num_str_values FROM distinct_values WHERE {prefix_condition} ORDER BY num_str_values DESC LIMIT 10;"
            with self.lock:
                cursor = self._conn.execute(prefix_query)
            prefix_matches = cursor.fetchall()

            num_prefix_matches = len(prefix_matches)
            self.logger.debug(f"Num prefix matches: {num_prefix_matches}")
            self.logger.debug(f"Prefix matches: {prefix_matches}")

            similarity_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS subset_distance, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS absolute_distance, ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS abs_length_difference, num_str_values FROM distinct_values WHERE LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) < 5 AND CAST(LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS DOUBLE) / NULLIF(LENGTH('{lower_literal}'), 0) < 0.3 ORDER BY subset_distance ASC, abs_length_difference ASC, num_str_values DESC LIMIT 2;"

            with self.lock:
                cursor = self._conn.execute(similarity_query)
            similarity_matches = cursor.fetchall()

            num_similarity_matches = len(similarity_matches)

            self.logger.debug(f"Num similarity matches: {num_similarity_matches}")
            self.logger.debug(f"Similarity matches: {similarity_matches}")

            if num_prefix_matches > 0 and num_prefix_matches <= 5:
                prefix_set = set()
                for prefix_match in prefix_matches:
                    prefix_set.add(prefix_match[0])
                num_similarity_prefix_overlaps = 0
                for similarity_match in similarity_matches:
                    if similarity_match[0] in prefix_set:
                        num_similarity_prefix_overlaps += 1
                self.logger.debug(f"Num Similarity Prefix overlaps: {num_similarity_prefix_overlaps}")
                if num_similarity_prefix_overlaps != 1:
                    if num_prefix_matches > 1:
                        # Multiple prefix matches, use string prefix
                        altered_literal["literal"] = f"{lower_literal_prefix}%"
                    else:
                        # Only one prefix match, use the full matched string
                        altered_literal["literal"] = prefix_matches[0][0]
                    if literal["operator"] in ("<>", "!=", "NOT LIKE", "NOT PG_ILIKE", "NOT ILIKE", "NOT PG_ILIKE"):
                        altered_literal["operator"] = "NOT ILIKE"
                    else:
                        altered_literal["operator"] = "ILIKE"
                    return altered_literal

            # If we are here, we do fuzzy similarity search
            if num_similarity_matches > 0:
                if (
                    num_similarity_matches > 1
                    and similarity_matches[0][1] == 0
                    and similarity_matches[0][1] == similarity_matches[1][1]
                ):

                    # Here there are at least two matches such that the user-provided literal is a full substring of
                    # the column value. In this case, we will match against all strings that our string literal
                    # is a substring of
                    altered_literal["literal"] = f"%{lower_literal}%"
                else:
                    # There was only one match, or two matches, so pick the
                    # top returned value. In the case of a tie, pick the prefix match if it exists (we've sorted in ascending order by score and descending order by number
                    if num_similarity_matches > 1 and similarity_matches[0][1] == similarity_matches[1][1]:

                        def matching_prefix_length(s1: str, s2: str) -> int:
                            match_length = 0
                            for c1, c2 in zip(s1, s2):
                                if c1 == c2:
                                    match_length += 1
                                else:
                                    break
                            return match_length

                        prefix_match_len_1 = matching_prefix_length(lower_literal, similarity_matches[0][0])
                        prefix_match_len_2 = matching_prefix_length(lower_literal, similarity_matches[1][0])
                        if prefix_match_len_1 >= prefix_match_len_2:
                            altered_literal["literal"] = str(similarity_matches[0][0])
                        else:
                            altered_literal["literal"] = str(similarity_matches[1][0])
                    else:
                        altered_literal["literal"] = str(similarity_matches[0][0])
            if literal["operator"] in ("<>", "!=", "NOT LIKE", "NOT PG_ILIKE", "NOT ILIKE", "NOT PG_ILIKE"):
                altered_literal["operator"] = "NOT ILIKE"
            else:
                altered_literal["operator"] = "ILIKE"
            return altered_literal

        return altered_literal

    def correct_string_literals(self, query: str, exact_match_threshold: float = 0.999999) -> str:
        config = get_config()
        if not config.enable_str_literal_correction:
            return query

        self.logger.debug(f"Correcting string literals in query: {query}")
        altered_query = deepcopy(query)
        try:
            str_literal_ops_list = self.get_string_literal_ops(query)
            if len(str_literal_ops_list) == 0:
                self.logger.debug("No string literals found in query")
                return query
            for str_literal_op in str_literal_ops_list:
                altered_str_literal_op = self.correct_string_literal(str_literal_op, exact_match_threshold)
                if altered_str_literal_op != str_literal_op:
                    if altered_str_literal_op["operator"] != str_literal_op["operator"]:
                        altered_query = altered_query.replace(
                            f"{str_literal_op['column']} {str_literal_op['operator']} '{str_literal_op['literal']}'",
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                        )
                        self.logger.debug(
                            f"Operator changed from {str_literal_op['operator']} to {altered_str_literal_op['operator']}"
                        )
                    if altered_str_literal_op["literal"] != str_literal_op["literal"]:
                        altered_query = altered_query.replace(
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{str_literal_op['literal']}'",
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                        )
                        self.logger.debug(
                            f"Literal changed from {str_literal_op['literal']} to {altered_str_literal_op['literal']}"
                        )
        except Exception as e:
            self.logger.info(f"Error correcting string literals: {e}")
            return query

        self.logger.debug(f"Altered query: {altered_query}")
        return altered_query

    def complexity(self, command: str) -> int:
        command = strip_sql_comments(command)
        plan = self.get_query_plan(command)
        return rate_sql_complexity(plan)

    def query_stats(self, command: str) -> dict[str, Any]:
        command = strip_sql_comments(command)
        plan = self.get_calcite_query_plan(command, detailed=False)
        return calc_query_stats(plan)

    def get_table_info_no_throw(self, table_names: Optional[list[str]] = None) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)  # type: ignore
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}"
