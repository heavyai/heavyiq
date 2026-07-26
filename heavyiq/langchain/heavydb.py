# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
import multiprocessing
import re
import weakref
from collections.abc import AsyncGenerator, Awaitable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from multiprocessing.managers import SyncManager
from threading import Lock
from typing import (Any, Callable, Iterator, NamedTuple, Optional, Sequence,
                    TypedDict)

import anyio
from async_lru import alru_cache
from heavydb import Connection as _HeavyDBConnection
from heavydb import Cursor
from heavydb._parsers import ColumnDetails, _extract_column_details
from heavydb.thrift.ttypes import TTableDetails
from starlette.concurrency import run_in_threadpool

from heavyiq.config import get_config
from heavyiq.langchain.heavydb_utils import DB_KEYWORDS
from heavyiq.utils import (LRUCache, calc_query_stats, is_destructive_sql,
                           rate_sql_complexity, strip_sql_comments)


class CustomColumnDetails(NamedTuple):
    """
    HeavyDB Table's custom column details including comment.
    """

    name: str
    type: str
    comment: str
    encoding: str
    is_array: bool
    is_text_column: Optional[bool] = None
    is_timestamp_column: Optional[bool] = None
    name_str: Optional[str] = None
    type_str: Optional[str] = None
    encoding_str: Optional[str] = None


class CustomTableDetails(NamedTuple):
    """
    HeavyDB Table's custom details including table and column comments.
    """

    name: str
    columns: list[CustomColumnDetails]
    comment: str


class Connection(_HeavyDBConnection):
    """
    pyheavydb's Connection plus the two convenience methods heavyai layered on top.

    These were the only reason HeavyIQ depended on heavyai. That package cannot be
    imported against pyheavydb 10 at all: heavyai.connection pulls in TCreateParams
    at module scope, and the struct went away with distributed mode.
    """

    def get_tables(self) -> list[str]:
        return self._client.get_tables(self._session)

    def get_column_details(self, table_name: str) -> list[ColumnDetails]:
        details = self._client.get_table_details(self._session, table_name)
        return _extract_column_details(details.row_desc)


def connect(uri: Optional[str] = None, **kwargs: Any) -> Connection:
    """heavydb.connect, but yielding the Connection subclass above."""
    return Connection(uri=uri, **kwargs)


class PersistantConnection(Connection):
    """
    Helps to establish heavydb connection which persists for atleast an hour.

    Default Connection gets closed once the corresponding __del__ method being called (ie. upon garbage collection).
    So we can't re-use the same session_id on `/query`, `/question` endpoints.

    This class helps to overcome the above issue, and the created session gets auto expire after certain limit
    (Default session-timeout-value set on server side is 60 mins).
    """

    def close(self):
        """Don't disconnect from the database. Let the session expire automatically."""
        self._closed = 1
        self._rbc = None


class StringLiteralOp(TypedDict):
    """
    A type representation for string literal operations.
    """

    operator: str
    literal: str
    database: str
    table: str
    column: str


class HeavyDB:
    """A heavydb database connection."""

    _manager: Optional[SyncManager] = None
    _top_k_cache: Optional[LRUCache[str, str]] = None
    _sample_rows_cache: Optional[LRUCache[str, str]] = None
    _table_schema_cache: Optional[LRUCache[str, str]] = None
    _table_text_columns_count_cache: Optional[LRUCache[str, int]] = None
    _table_total_row_count_cache: Optional[LRUCache[str, int]] = None
    _timestamp_cache: Optional[LRUCache[str, str]] = None

    def __init__(
        self,
        conn: Connection,
        ignore_tables: Optional[list[str] | tuple[str]] = None,
        include_tables: Optional[list[str] | tuple[str]] = None,
        sample_rows_in_table_info: int = 2,
        custom_table_info: Optional[dict[str, str]] = None,
    ):
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        from heavyiq.logging_utils import get_heavyiq_logger

        self.logger = get_heavyiq_logger()
        self._conn = conn
        self.lock = Lock()
        self.alock = asyncio.Lock()
        self._dbname = self._conn._dbname
        self._table_schema_change_callback: Callable[["HeavyDB", str], None] | None = (
            None  # callback which deals with the table schema change
        )
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

        # cleanup at the last
        weakref.finalize(self, self.cleanup)

    @property
    def table_schema_change_callback(self) -> Callable:
        return self._table_schema_change_callback

    @table_schema_change_callback.setter
    def table_schema_change_callback(self, callback: Callable | None):
        self._table_schema_change_callback = callback

    def cleanup(self):
        try:
            with self.lock:
                self._conn.close()
        except Exception as e:
            print(f"Error on HeavyDB cleanup: {e}")

    @classmethod
    def get_manager(cls: type[HeavyDB]) -> SyncManager:
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
        return cls._manager

    @classmethod
    def initialize(cls: type[HeavyDB]) -> None:
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
    def get_top_k_cache(cls: type[HeavyDB]) -> LRUCache[str, str]:
        if cls._top_k_cache is None:
            cls._top_k_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._top_k_cache

    @classmethod
    def get_sample_rows_cache(cls: type[HeavyDB]) -> LRUCache[str, str]:
        if cls._sample_rows_cache is None:
            cls._sample_rows_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._sample_rows_cache

    @classmethod
    def get_table_schema_cache(cls: type[HeavyDB]) -> LRUCache[str, str]:
        if cls._table_schema_cache is None:
            cls._table_schema_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._table_schema_cache

    @classmethod
    def get_table_text_columns_count_cache(cls: type[HeavyDB]) -> LRUCache[str, int]:
        if cls._table_text_columns_count_cache is None:
            cls._table_text_columns_count_cache = LRUCache[str, int](manager=cls.get_manager())
        return cls._table_text_columns_count_cache

    @classmethod
    def get_table_total_row_count_cache(cls: type[HeavyDB]) -> LRUCache[str, int]:
        if cls._table_total_row_count_cache is None:
            cls._table_total_row_count_cache = LRUCache[str, int](manager=cls.get_manager())
        return cls._table_total_row_count_cache

    @classmethod
    def get_timestamp_cache(cls: type[HeavyDB]) -> LRUCache[str, str]:
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

    @classmethod
    def _connect_with_timeout(
        cls: type[HeavyDB], connect_func: Callable[[], Connection], timeout: float = 10
    ) -> Connection:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(connect_func)
            try:
                return future.result(timeout)
            except TimeoutError:
                raise Exception(f"HeavyDB failed to connect after {timeout} seconds")

    @classmethod
    async def _aconnect_with_timeout(
        cls: type[HeavyDB], coroutine: Awaitable[Connection], timeout: float = 10
    ) -> Connection:
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
    async def from_env_async(cls: type[HeavyDB], db_name: Optional[str] = None, **kwargs: Any) -> HeavyDB:
        """Create a database connection from environment variables."""
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        async def aconnect_func() -> Connection:
            func = functools.partial(
                connect,
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=db_name or config.heavydb_dbname,
                protocol=config.heavydb_protocol,
            )
            return await anyio.to_thread.run_sync(func, cancellable=True)

        conn = await cls._aconnect_with_timeout(aconnect_func(), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_uri(cls: type[HeavyDB], database_uri: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a database URI."""
        conn = cls._connect_with_timeout(lambda: connect(database_uri), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_env(cls: type[HeavyDB], db_name: Optional[str] = None, **kwargs: Any) -> HeavyDB:
        """Create a database connection from environment variables."""
        config = get_config()
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
        return cls(conn, **kwargs)

    @classmethod
    def from_session(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a session id."""
        config = get_config()

        def connect_func() -> Connection:
            return connect(
                sessionid=session_id,
                host=config.heavydb_host,
                port=config.heavydb_port,
                protocol=config.heavydb_protocol,
            )

        conn = cls._connect_with_timeout(connect_func, kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    @alru_cache(maxsize=128, ttl=60 * 5)
    async def _from_session_async_cached(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """
        Creates a current class instance (which further establishes heavydb connection)
        from the passed session_id only if the relevant instance for the passed session_id
        does not exists in the cache else it returns the cached instance.
        """
        config = get_config()

        async def aconnect_func() -> Connection:
            func = functools.partial(
                connect,
                sessionid=session_id,
                host=config.heavydb_host,
                port=config.heavydb_port,
                protocol=config.heavydb_protocol,
            )
            return await anyio.to_thread.run_sync(func, cancellable=True)  # type: ignore

        conn = await cls._aconnect_with_timeout(aconnect_func(), kwargs.pop("timeout", 10))
        instance = await run_in_threadpool(cls, conn, **kwargs)
        return weakref.proxy(instance)  # Cache a weak reference (helps to avoid memory leaks)

    @classmethod
    async def from_session_async(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """
        Always use this corountine function for instance creation since it implements caching.
        Create a database connection from session.
        Falls back to the original function if the weak reference is invalid.
        """
        copied_kwargs = kwargs.copy()
        new_kwargs = {}
        for key, value in copied_kwargs.items():
            if isinstance(value, list):
                new_kwargs[key] = tuple(value)
            else:
                new_kwargs[key] = value
        try:
            # Try to get the cached instance
            instance = await cls._from_session_async_cached(session_id, **new_kwargs)
            # try to access it's attributes in-order to check whether the original instance gets deleted or not, if yes then it should raise ReferenceError
            _ = instance._conn
            return instance
        except ReferenceError:
            # If weak reference is invalid, recreate the instance and update the cache
            print(f"Recreating HeavyDB instance for session_id: {session_id} due to ReferenceError")
            cls._from_session_async_cached.cache_clear()  # Clear the invalid entry from the cache
            instance = await cls._from_session_async_cached(session_id, **new_kwargs)
            return instance

    @classmethod
    def from_creds(
        cls: type[HeavyDB], username: str, password: str, dbname: Optional[str] = None, **kwargs: Any
    ) -> HeavyDB:
        """Create a database connection from username and password."""
        config = get_config()
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
        return cls(conn, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return "ANSI SQL"

    async def aexecute(self, query: str) -> Cursor:
        """
        Executes a SQL query on the HeavyDB database asynchronously and returns a cursor to the result set.

        This method ensures thread safety by acquiring an asynchronous lock before executing the query.
        The actual execution is offloaded to a thread pool to avoid blocking the event loop.

        Parameters:
            query (str): The SQL query to be executed on the HeavyDB database.

        Returns:
            Cursor: A cursor object that can be used to fetch the results of the query.
        """
        async with self.alock:
            return await run_in_threadpool(self._conn.execute, query)

    def get_usable_table_names(self) -> set[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    @classmethod
    def create_session_id(cls: type["HeavyDB"], db_name: str | None = None) -> str:
        """
        Create a new db session and return it's id.
        """
        config = get_config()
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
    async def create_with_persistant_connection_async(
        cls: type[HeavyDB], db_name: Optional[str] = None, **kwargs: Any
    ) -> HeavyDB:
        """
        Creates a HeavyDB instance with a persistant db connection.
        """
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        async def aconnect_func() -> PersistantConnection:
            func = functools.partial(
                PersistantConnection,
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=db_name or config.heavydb_dbname,
                protocol=config.heavydb_protocol,
            )
            return await anyio.to_thread.run_sync(func, cancellable=True)

        conn = await cls._aconnect_with_timeout(aconnect_func(), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    async def create_session_id_async(cls: type[HeavyDB], db_name: Optional[str] = None, **kwargs: Any) -> str:
        """Creates a persistant database connection from environment variables and returns it's session id."""
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        async def aconnect_func() -> PersistantConnection:
            func = functools.partial(
                PersistantConnection,
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=db_name or config.heavydb_dbname,
                protocol=config.heavydb_protocol,
            )
            return await anyio.to_thread.run_sync(func, cancellable=True)

        conn = await cls._aconnect_with_timeout(aconnect_func(), kwargs.pop("timeout", 10))
        return conn._session

    @property
    def table_info(self) -> str:
        """Information about all usable tables in the database."""
        return self.get_table_info()

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

    async def aget_column_timestamp(self, table_name: str, column: str) -> tuple[str, str]:
        """
        Get min/max values for a particular timestamp column.
        """
        self.logger.debug(f"Getting timestamp values for column {column} in table {table_name}")
        min_max_statement = f'SELECT min("{column}"), max("{column}") FROM "{table_name}";'
        cursor = await self.aexecute(min_max_statement)
        min_value, max_value = [str(v) for v in cursor.fetchone()]
        self.logger.debug(f"Got timestamp values for column {column} in table {table_name}")
        return min_value, max_value

    async def _aget_timestamp(self, table_name: str, timestamp_columns: Sequence[str]) -> Iterator[str, tuple]:
        """
        Cached version of aget_timestamp.
        """
        self.logger.debug(f"Getting min/max timestamp values for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        timestamp_values = self.timestamp_cache.get(cache_key)
        if timestamp_values is not None:
            self.logger.info(f"Got timestamp values for table {table_name} from cache")
        else:
            if not timestamp_columns:
                return []
            self.logger.info(f"Calculating timestamp values for table {table_name}")
            timestamp_values = await asyncio.gather(
                *[self.aget_column_timestamp(table_name, col) for col in timestamp_columns]
            )
            self.timestamp_cache.put(cache_key, timestamp_values)

        return zip(timestamp_columns, timestamp_values)

    async def aget_timestamp(self, table_name: str) -> str:
        """
        Get min/max values for all the timestamp, data columns exists on a table.
        """
        timestamp_columns = [
            c.name
            for c in self.get_table_columns(table_name)
            if c.type in ["TIMESTAMP", "DATE"] and c.is_array is False
        ]
        timestamp_col_str = ""

        if timestamp_columns:
            zip_column_timestamp = await self._aget_timestamp(
                table_name=table_name, timestamp_columns=timestamp_columns
            )

            timestamp_col_str += "Timestamp and date columns with min and max values:\n"
            for col, (min_value, max_value) in zip_column_timestamp:
                timestamp_col_str += f"{col}: ({min_value}, {max_value})\n"

        return timestamp_col_str

    async def aget_query_plan(self, query: str) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        cursor = await self.aexecute(f"EXPLAIN PLAN {query}")
        result: tuple[str] = cursor.fetchone()  # type: ignore
        return str(result[0])

    async def acomplexity(self, command: str) -> int:
        try:
            self.logger.debug("Calculation SQL query complexity!")
            command = strip_sql_comments(command)
            plan = await self.aget_query_plan(command)
            out = rate_sql_complexity(plan)
            self.logger.debug("Successfully calculated SQL query complexity!")
            return out
        except Exception as e:
            self.logger.error(f"Failed to calculate complexity, {e}")
            return 0

    async def aquery_stats(self, command: str) -> dict[str, int]:
        command = strip_sql_comments(command)
        plan = await self.aget_calcite_query_plan(command, detailed=False)
        return calc_query_stats(plan)

    async def avalidate_query(self, query: str) -> list:
        """Validate a query."""
        # Remove block comments
        self.logger.debug("Validating SQL query!")
        try:
            query = strip_sql_comments(query)
            if is_destructive_sql(query):
                raise ValueError("Destructive SQL is not allowed")
            if "::" in query:
                raise ValueError("Double colon cast syntax is not allowed. Use CAST() instead.")
            async with self.alock:
                out = await run_in_threadpool(self._conn._client.sql_validate, self._conn._session, query)
        except Exception as e:
            self.logger.exception("SQL query validation failed!")
            raise e
        else:
            self.logger.debug("Successfully completed SQL query validation.")
            return out

    async def aget_column_top_k(self, table: str, column: str, _k: int = 5) -> tuple[Optional[list[str]], bool]:
        config = get_config()
        cardinality_threshold = config.column_top_k_cardinality_threshold
        high_cardinality_sample = config.column_top_k_high_cardinality_sample

        # always quote all the columns
        column = f'"{column}"'
        row_count = await self.aget_total_row_count(table_name=table)
        if row_count > config.column_top_k_max_unique_values_count:
            # check whether the number of unique values of this column is greather than certain threshold value or not
            # if yes, then just return the few values from sample else calculate top-k
            unique_values_stmt = f'SELECT APPROX_COUNT_DISTINCT({column}) FROM "{table}";'
            cursor = await self.aexecute(unique_values_stmt)
            unique_values_count = int(cursor.fetchone()[0])
            if ugtmax := (unique_values_count > config.column_top_k_max_unique_values_count) or (
                (unique_values_count / row_count) * 100 >= 95.0
            ):
                self.logger.debug(
                    f"Column {column} distinct values exceeds the threshold. Returning sample {high_cardinality_sample} values"
                    if ugtmax
                    else f"Column {column} distinct values count almost close to the total row count. Returning sample {high_cardinality_sample} values"
                )
                # find values upto a certain limit from sample
                cursor = await self.aexecute(
                    f'SELECT {column} FROM "{table}" WHERE {column} IS NOT NULL LIMIT {high_cardinality_sample};'
                )
                top_k_res: list[str] = [str(v[0]).replace("\n", " ") for v in cursor.fetchall()]
                return top_k_res, True

        # check to see if the column is low cardinality
        # fetch top (threshold + 1)
        # if there are < (threshold + 1) values, it's low cardinality and we can return all of them
        # if there are >= (threshold + 1) values, it's high cardinality and we need to sample the top high_cardinality_sample
        """Get the top k values for a column."""
        self.logger.debug(f"Getting top k values for column {column} in table {table}")
        top_k_statement = f'SELECT {column}, COUNT(*) as cnt FROM "{table}" WHERE {column} is not null GROUP BY {column} ORDER BY cnt DESC LIMIT {cardinality_threshold + 1};'
        cursor = await self.aexecute(top_k_statement)
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

    async def aget_table_columns(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        self.logger.debug(f"Getting columns for table {table}")
        column_details = await self._aget_column_details(table)
        self.logger.debug(f"Got columns for table {table}")
        return column_details

    async def should_refresh_table_cache(self, table: str) -> bool:
        """
        Determines whether the cache for a specified table should be refreshed.

        Args:
            table (str): The name of the table for which the cache status is checked.

        Returns:
            bool: True if the table cache should be refreshed, False otherwise.
        """
        self.logger.info(f"Checking for {table} table cache refresh...")
        cache_key = f"{self._dbname}.{table}"
        keys_found = self.table_schema_cache.get_key_starts_with(cache_key)

        # compare the LRU cache value stored for a particular table with the
        # async_lru cache value (shorter cache) for the same table
        most_recent_key_args = None
        if keys_found:
            most_recent_key = None
            for key in keys_found[::-1]:
                cached_schema = self.table_schema_cache.get(key)
                if cached_schema:
                    self.logger.info(f"Found a most recent key for : {table}")
                    most_recent_key = key
                    break

            if most_recent_key is None:
                self.logger.info(f"most recent key is none : {table}")
                return False

            _, _, x, y, z = most_recent_key.split(".")
            most_recent_key_args = tuple([True if i == "True" else False for i in [x, y, z]])
        else:
            # no keys found to compare, so return False
            self.logger.info(f"no keys found for table : {table}")
            return False

        # check for any diff in current schema and cached schema
        # if yes then return True else return False
        if most_recent_key_args:
            current_schema = await self._aget_raw_table_schema_from_thrift(table, *most_recent_key_args)
        else:
            current_schema = await self._aget_raw_table_schema_from_thrift(table)

        if cached_schema == current_schema:
            self.logger.info(f"Table {table} schema unchanged.")
            return False

        self.logger.info(f"Table {table} schema changed, invalidating table caches...")
        return True

    async def delete_table_cache(self, table: str) -> None:
        """
        Deletes all the cache entries associated with a particular table which includes top-k, sample_rows, schema, etc.
        """
        self.logger.info(f"Deleteing all caches for the table {table}")
        cache_key = f"{self._dbname}.{table}"
        self.table_schema_cache.delete_by_key_prefix(cache_key)
        self.top_k_cache.delete(cache_key)
        self.sample_rows_cache.delete(cache_key)
        self.table_text_columns_count_cache.delete(cache_key)
        self.table_total_row_count_cache.delete(cache_key)
        self.timestamp_cache.delete(cache_key)
        # invalidate async-lru caches
        self._aget_table_details.cache_invalidate(table)
        self._aget_column_details.cache_invalidate(table)
        self.aget_text_columns.cache_invalidate(table)

    async def trigger_table_schema_change_callback(self, table: str, callback: Callable | None = None):
        schema_change_callback = callback or self.table_schema_change_callback
        if schema_change_callback:
            # run the callback as background task
            self.logger.debug(f"Schema change detected, callback initiated for {table} table.")
            asyncio.create_task(schema_change_callback(self._conn._session, table))

    async def check_and_invalidate_table_cache(self, table: str) -> None:
        """
        Checks for table schema changes and invalidates the table caches accordingly.
        It also triggers the callback with the corresponding table name.
        """
        if await self.should_refresh_table_cache(table):
            await self.delete_table_cache(table)
            await self.trigger_table_schema_change_callback(table)

    @alru_cache(
        maxsize=32, ttl=10
    )  # [Obselete] internal caches are used since this method gets called atleast 2 times in a single request
    async def _aget_raw_table_schema(self, table: str) -> str:
        """
        Gets the table schema from db only.
        """
        create_command = f'SHOW CREATE TABLE "{table}";'

        cursor = await self.aexecute(create_command)

        table_schema = cursor.fetchone()[0]  # type: ignore
        table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)  # type: ignore
        table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
        table_schema = re.sub(r"\n", "", table_schema)
        table_schema = re.sub(r"\( +", "(", table_schema)
        table_schema = re.sub(r", +", ", ", table_schema)
        if "WITH (" in table_schema:
            table_schema = table_schema[: table_schema.index("WITH (")]

        return table_schema

    @alru_cache(ttl=60 * 10)  # store atleast for 10 mins
    async def _aget_table_details(self, table: str) -> TTableDetails:
        """
        Get table details through thrift endpoint.
        """
        async with self.alock:
            table_details = await run_in_threadpool(self._conn.get_table_details, table)
        return table_details

    @alru_cache(ttl=60)  # store atleast for 10 secs
    async def _aget_column_details(self, table: str) -> list[ColumnDetails]:
        """
        Get table column details through thrift endpoint.
        """
        async with self.alock:
            column_details = await run_in_threadpool(self._conn.get_column_details, table)
        return column_details

    async def _aget_table_custom_details(self, table: str, use_cache: bool = True) -> CustomTableDetails:
        """
        Return custom details of a heavyDB table.
        """
        if use_cache:
            table_details = await self._aget_table_details(table)
        else:
            async with self.alock:
                table_details = await run_in_threadpool(self._conn.get_table_details, table)
        table_comment = table_details.comment or ""
        columns: list[ColumnDetails] = _extract_column_details(table_details.row_desc)
        column_name_comments_mapping = {x.col_name: x.comment or "" for x in table_details.row_desc}
        custom_columns = [
            CustomColumnDetails(
                name=col.name,
                name_str=f'"{col.name}"' if col.name.upper() in DB_KEYWORDS else col.name,
                type=col.type,
                type_str="TEXT" if col.type == "STR" else col.type,
                comment=column_name_comments_mapping[col.name],
                encoding=col.encoding,
                encoding_str=(
                    col.encoding if (col.type == "STR" and col.encoding == "NONE") else ""
                ),  # add NONE encoding for only text columns
                is_array=col.is_array,
                is_timestamp_column=True if (col.type in ["TIMESTAMP", "DATE"] and col.is_array is False) else False,
                is_text_column=(
                    True if (col.type == "STR" and col.encoding == "DICT" and col.is_array is False) else False
                ),
            )
            for col in columns
        ]
        return CustomTableDetails(name=table, columns=custom_columns, comment=table_comment)

    async def _aget_raw_table_schema_from_thrift(
        self,
        table: str,
        include_top_k: bool = True,
        include_timestamp: bool = True,
        include_comments: bool = True,
        use_cache: bool = True,
    ) -> str:
        """
        Method used to form table_schema from `get_table_details` thrift endpoint without re-using any cache value unless use_cache is False.
        use_cache=True, cached values of top-k and timestamp values are being used.
        """
        table_details: CustomTableDetails = await self._aget_table_custom_details(table, use_cache=use_cache)
        config = get_config()
        table_name = f'"{table_details.name}"' if table_details.name.upper() in DB_KEYWORDS else f"{table_details.name}"
        schema_stmt = (
            "CREATE TABLE {table_name} /* {table_comment} */ (\n{column_details});"
            if (include_comments and table_details.comment)
            else "CREATE TABLE {table_name} {table_comment}(\n{column_details});"
        )
        column_metadata_mapping = {}
        text_columns, timestamp_columns = [], []
        for col in table_details.columns:
            if col.is_text_column:
                text_columns.append(col.name)
            elif col.is_timestamp_column:
                timestamp_columns.append(col.name)

        if include_top_k and text_columns:
            # retrieve from top-k cache or calculate
            if use_cache:
                zip_colname_topk = await self._aget_top_k(table_name=table, text_columns=text_columns)
            else:
                columns_top_k = await asyncio.gather(
                    *[self.aget_column_top_k(table_details.name, col) for col in text_columns]
                )
                zip_colname_topk = zip(text_columns, columns_top_k)

            for colstr, (top_k_res, is_high_cardinality) in zip_colname_topk:
                if top_k_res and is_high_cardinality:
                    column_metadata_mapping[colstr] = top_k_res[:-1] + [top_k_res[-1] + " ..."]
                elif top_k_res:
                    column_metadata_mapping[colstr] = top_k_res

        if include_timestamp and timestamp_columns:
            if use_cache:
                zip_column_timestamp = await self._aget_timestamp(table_name=table, timestamp_columns=timestamp_columns)
            else:
                timestamp_values = await asyncio.gather(
                    *[self.aget_column_timestamp(table_details.name, col) for col in timestamp_columns]
                )
                zip_column_timestamp = zip(timestamp_columns, timestamp_values)

            for colstr, (min_value, max_value) in zip_column_timestamp:
                column_metadata_mapping[colstr] = [min_value, max_value]

        def format_column(column: CustomColumnDetails) -> str:
            """
            Formats column for prompt.
            """
            parts = [column.name_str]
            if column.is_array:
                parts.append(f"{column.type_str}[]")
            else:
                parts.append(column.type_str)
            encoding = column.encoding_str
            if encoding:
                parts.append(f"ENCODING {encoding}")

            # add column metadata (top-k, time-range) inline.
            col_metadata = column_metadata_mapping.get(column.name)
            if col_metadata:
                parts.append("({})".format(", ".join(col_metadata)))

            if include_comments:
                comment = f"/* {column.comment} */" if column.comment else ""
                if comment:
                    parts.append(comment)

            return " ".join(parts)

        table_comment = (table_details.comment or "") if include_comments else ""
        return schema_stmt.format(
            table_name=table_name,
            table_comment=table_comment,
            column_details=config.inline_column_metadata_delimiter.join(
                [format_column(x) for x in table_details.columns]
            ),
        )

    async def aget_table_schema(
        self, table: str, include_top_k: bool = True, include_timestamp: bool = True, include_comments: bool = True
    ) -> str:
        """
        Get table schema from cache for from db async.
        """
        self.logger.debug(f"Getting schema for table {table}")
        cache_key = f"{self._dbname}.{table}.{include_top_k}.{include_timestamp}.{include_comments}"
        cached_value = self.table_schema_cache.get(cache_key)
        if cached_value is not None:
            self.logger.info(f"cache key -> {cache_key}")
            self.logger.info(f"Got schema for table {table} from cache")
            return cached_value
        # Get table schema from thrift endpoint
        self.logger.info(f"Fetching the schema for {table} table")
        table_schema = await self._aget_raw_table_schema_from_thrift(
            table, include_top_k=include_top_k, include_timestamp=include_timestamp, include_comments=include_comments
        )
        self.table_schema_cache.put(cache_key, table_schema)
        self.logger.debug(f"Got schema for table {table}")

        return table_schema

    async def aget_sample_rows(self, table_name: str) -> str:
        """
        Get table sample rows from cache for db async.
        """
        self.logger.debug(f"Getting sample rows for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        cached_value = self.sample_rows_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got sample rows for table {table_name} from cache")
            return cached_value
        # build the select command
        command = f'SELECT * FROM "{table_name}" LIMIT {self._sample_rows_in_table_info};'

        # save the columns in string format
        columns_str = ",".join([col.name for col in await self.aget_table_columns(table_name)])

        # get the sample rows
        cursor = await self.aexecute(command)
        # shorten values in the sample rows
        sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], cursor))

        # save the sample rows in string format
        sample_rows_str = "\n".join([",".join(row) for row in sample_rows])

        res = f"{self._sample_rows_in_table_info} rows from {table_name} table:\n{columns_str}\n{sample_rows_str}"

        self.sample_rows_cache.put(cache_key, res)
        self.logger.debug(f"Got sample rows for table {table_name}")
        return res

    @alru_cache(ttl=10)  # ttl of 10 secs alone would be enough to set row_count value in shared inter-process cache
    async def aget_total_row_count(self, table_name: str) -> int:
        """
        Get total row count of the given database table.
        """
        cache_key = f"{self._dbname}.{table_name}"
        cached_value = self.table_total_row_count_cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        # build the select command
        command = f'SELECT COUNT(*) FROM "{table_name}"'
        # Get the total row count
        cursor = await self.aexecute(command)
        row_count = cursor.fetchone()[0]
        self.table_total_row_count_cache.put(cache_key, row_count)
        return row_count

    @alru_cache(maxsize=32, ttl=10)  # internal cache used to cache the text columns for 10 secs
    async def aget_text_columns(self, table_name: str) -> list[str]:
        """
        Retrieve the list of text columns available in a table.
        """
        return [
            c.name
            for c in await self.aget_table_columns(table_name)
            if c.type == "STR" and c.encoding == "DICT" and c.is_array is False
        ]

    async def aget_text_columns_count(self, table_name: str) -> int:
        """
        Returns the number of text columns available in a table.
        """
        self.logger.debug(f"Getting text columns count for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        cached_value = self.table_text_columns_count_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got text columns count for table {table_name} from cache")
            return cached_value

        text_columns = await self.aget_text_columns(table_name)
        text_columns_count = len(text_columns)
        self.table_text_columns_count_cache.put(cache_key, text_columns_count)
        self.logger.debug(f"Got text columns count for table {table_name}")
        return text_columns_count

    async def _aget_top_k(self, table_name: str, text_columns: Sequence[str]) -> Iterator[str, tuple]:
        """
        Get or find top-k for each text columns, save it in cache, return it as zip or text-columns, top-k mapping.
        """
        self.logger.debug(f"Getting top k values for table {table_name}")
        cache_key = f"{self._dbname}.{table_name}"
        columns_top_k = self.top_k_cache.get(cache_key)

        if columns_top_k is not None:
            self.logger.info(f"Got top k values for table {table_name} from cache")
        else:
            if not text_columns:
                return []
            self.logger.info(f"Calculating top k values for table {table_name}")
            columns_top_k = await asyncio.gather(*[self.aget_column_top_k(table_name, col) for col in text_columns])
            self.top_k_cache.put(cache_key, columns_top_k)

        return zip(text_columns, columns_top_k)

    async def aget_top_k(self, table_name: str) -> str:
        """
        Get table top k rows from cache for db async.
        """
        low_cardinality_columns = []
        high_cardinality_columns = []
        text_columns = await self.aget_text_columns(table_name)
        zip_colname_topk = await self._aget_top_k(table_name=table_name, text_columns=text_columns)
        for col, (top_k_res, is_high_cardinality) in zip_colname_topk:
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

        return top_k_strings

    async def aget_single_table_info(
        self,
        table_name: str,
        include_samples: bool = True,
        include_top_k: bool = True,
        include_timestamp: bool = True,
        include_comments: bool = True,
    ) -> str:
        """Get information about a single table asynchronously."""
        all_table_names = self.get_usable_table_names()
        if table_name not in all_table_names:
            raise ValueError(f"table_name {table_name} not found in database")

        if self._custom_table_info and table_name in self._custom_table_info:
            return self._custom_table_info[table_name]

        # It's not the right method to invoke table schema change callback through check_and_invalidate_table_cache method
        # since aget_single_table_info can be called many times upon prompt generation in-order to get the desired table_info
        # within the context length
        # await self.check_and_invalidate_table_cache(table_name)
        config = get_config()
        tasks = []
        # if inline column metadata config enabled then
        # let the get_table_schema method to return table schema along with top-k and timestamp text
        # else make the method to return only schema long with optional comments
        if config.inline_column_metadata_on_table_info_prompt:
            tasks.append(
                self.aget_table_schema(
                    table_name,
                    include_top_k=include_top_k,
                    include_timestamp=include_timestamp,
                    include_comments=include_comments,
                )
            )
        else:
            tasks.append(
                self.aget_table_schema(
                    table_name, include_top_k=False, include_timestamp=False, include_comments=include_comments
                )
            )
            # since top-k, timestamp data not included on the table_schema text
            # we have to detect it manually by calling it's relevant functions
            if include_top_k:
                tasks.append(self.aget_top_k(table_name))
            if include_timestamp:
                # include timestamp,date columns with min/max values
                tasks.append(self.aget_timestamp(table_name))

        if include_samples and self._sample_rows_in_table_info:
            tasks.append(self.aget_sample_rows(table_name))

        results = await asyncio.gather(*tasks)
        return "\n".join(results).strip()

    async def aget_table_info(self, table_names: Optional[list[str]] = None, **kwargs) -> str:
        """
        Get information about specified tables asyc.
        """
        config = get_config()
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            table_names_set = set(table_names)
            missing_tables = table_names_set.difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names_set

        if len(all_table_names) > 1 and config.sort_prompt_tables_desc:
            # sort tables in desc order based on total row count
            table_row_count = await asyncio.gather(*[self.aget_total_row_count(table) for table in all_table_names])
            sorted_tables = [
                k[0] for k in sorted(zip(all_table_names, table_row_count), key=lambda x: x[1], reverse=True)
            ]
        else:
            sorted_tables = list(all_table_names)

        tables = []
        for table in sorted_tables:
            tables.append(self.aget_single_table_info(table, **kwargs))

        tasks_output = await asyncio.gather(*tables)

        final_str = "\n\n".join([i.strip() for i in tasks_output])
        return final_str

    async def arun(self, command: str, fetch: str = "all", to_str: bool = True) -> str | tuple | list:
        """
        Execute a SQL command and return a string representing the results async.
        """
        command = strip_sql_comments(command)
        if is_destructive_sql(command):
            raise ValueError("Destructive SQL is not allowed")
        cursor = await self.aexecute(command)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")
        if to_str:
            return str(result)
        return result  # type: ignore

    async def aget_calcite_query_plan(self, query: str, detailed: bool = False) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        sql_stmt = f"EXPLAIN CALCITE DETAILED {query}" if detailed else f"EXPLAIN CALCITE {query}"
        cursor = await self.aexecute(sql_stmt)
        query_plan: tuple[str] = cursor.fetchone()  # type: ignore
        return str(query_plan[0])

    async def aextract_column_mappings(self, detailed_query_plan: str) -> dict[str, tuple[str, str, str]]:
        self.logger.debug(f"Extracting column mappings from query plan: {detailed_query_plan}")
        result = {}
        # Use a regex pattern to capture (database, table, column) and literal values in LogicalFilter
        pattern = r"\[\$([0-9]+)->db:([\w]+),tableName:([\w]+),colName:([\w]+)\]"
        matches = re.findall(pattern, detailed_query_plan)
        for id, db, table, column in matches:
            col_id = (db, table, column)
            result[id] = col_id

        self.logger.debug(f"Extracted column mappings: {result}")
        return result

    async def aextract_string_literal_ops(self, detailed_query_plan: str) -> dict[str, tuple[str, str]]:
        self.logger.debug(f"Extracting string literal operations from query plan: {detailed_query_plan}")
        result = {}
        pattern1 = r"(NOT\()?(LIKE|PG_ILIKE|>=|<=|<>|=)\(\$(\d+),\s*'((?:''|[^'])+)'\)"
        pattern2 = r"(NOT\()?(LIKE|PG_ILIKE|>=|<=|<>|=)\('((?:''|[^'])+)',\s*\$(\d+)\)"

        matches1 = re.findall(pattern1, detailed_query_plan)
        matches2 = re.findall(pattern2, detailed_query_plan)
        for negation, op, id, literal in matches1:
            if negation == "NOT(":
                op = f"NOT {op}"
            result[id] = (op, literal)
        for negation, op, literal, id in matches2:
            if negation == "NOT(":
                op = f"NOT {op}"
            result[id] = (op, literal)

        self.logger.debug(f"Extracted string literal operations: {result}")
        return result

    async def aget_string_literal_ops(self, query: str) -> list[StringLiteralOp]:
        detailed_query_plan = await self.aget_calcite_query_plan(query, detailed=True)
        col_mapping, str_literal_ops = await asyncio.gather(
            self.aextract_column_mappings(detailed_query_plan), self.aextract_string_literal_ops(detailed_query_plan)
        )
        result: list[StringLiteralOp] = []
        for id, (op, literal) in str_literal_ops.items():
            db, table, column = col_mapping[id]
            result.append({"operator": op, "literal": literal, "database": db, "table": table, "column": column})
        return result

    async def aretrieve_columns_from_query(self, query: str) -> dict[str, tuple[str, str, str]]:
        """
        Gets the columns used in the input SQL query.
        """
        detailed_query_plan = await self.aget_calcite_query_plan(query, detailed=True)
        column_mapping = await self.aextract_column_mappings(detailed_query_plan)
        return column_mapping

    async def acorrect_string_literal(self, literal: StringLiteralOp, exact_match_threshold: float) -> StringLiteralOp:
        """
        Tries to correct a given string literal by searching for close matches in a database async.
        """
        self.logger.debug(f"Correcting string literal: {literal['column']} : {literal['literal']}")
        case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE '{literal['literal']}' GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
        cursor = await self.aexecute(case_match_query)
        case_match_rows = cursor.fetchall()

        num_case_match_rows = len(case_match_rows)
        exact_match_count = 0
        total_count = 0
        altered_literal = deepcopy(literal)

        for row in case_match_rows:
            if row[0] == literal["literal"]:
                exact_match_count += row[1]  # type: ignore
            total_count += row[1]  # type: ignore

        if total_count > 0:  # and literal["operator"] != "ILIKE":
            if exact_match_count == 0 and num_case_match_rows == 1:
                matched_literal = str(case_match_rows[0][0])
                # if ' exists in matched rows, then one or more ' with exactly two single quotes
                if "'" in matched_literal:
                    matched_literal = re.sub(r"'+", "''", matched_literal)
                altered_literal["literal"] = matched_literal
                return altered_literal
            elif exact_match_count / total_count < exact_match_threshold:
                if literal["operator"] in ("<>", "!=", "NOT LIKE", "NOT PG_ILIKE", "NOT ILIKE", "NOT PG_ILIKE"):
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

            cursor = await self.aexecute(prefix_query)
            prefix_matches = cursor.fetchall()

            num_prefix_matches = len(prefix_matches)
            self.logger.debug(f"Num prefix matches: {num_prefix_matches}")

            similarity_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS subset_distance, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS absolute_distance, ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) AS abs_length_difference, num_str_values FROM distinct_values WHERE LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) < 5 AND CAST(LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') AS DOUBLE) / NULLIF(LENGTH('{lower_literal}'), 0) < 0.3 ORDER BY subset_distance ASC, abs_length_difference ASC, num_str_values DESC LIMIT 2;"
            cursor = await self.aexecute(similarity_query)
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
                        # correct the prefix_match before adding it to the altered query
                        # since it's a result of another query
                        altered_literal["literal"] = re.sub(r"'+", "''", prefix_matches[0][0])
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

                        def matching_prefix_length(s1, s2):
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

    async def acorrect_string_literals(self, query: str, exact_match_threshold: float = 0.999999) -> str:
        config = get_config()
        if not config.enable_str_literal_correction:
            return query

        self.logger.debug(f"Correcting string literals in query: {query}")
        altered_query = query
        try:
            str_literal_ops_list = await self.aget_string_literal_ops(query)
            if len(str_literal_ops_list) == 0:
                self.logger.debug("No string literals found in query")
                return query
            for str_literal_op in str_literal_ops_list:
                altered_str_literal_op = await self.acorrect_string_literal(str_literal_op, exact_match_threshold)
                if altered_str_literal_op != str_literal_op:
                    if (
                        altered_str_literal_op["operator"] != str_literal_op["operator"]
                        or altered_str_literal_op["literal"] != str_literal_op["literal"]
                    ):
                        search_literal_op = str_literal_op["operator"]
                        if search_literal_op == "<>" or search_literal_op == "!=":
                            # Handle case where AST shows <> but string is !=
                            search_literal_op = "(<>|!=)"
                        elif search_literal_op == "ILIKE" or search_literal_op == "PG_ILIKE":
                            search_literal_op = "(ILIKE|PG_ILIKE)"
                        elif search_literal_op == "NOT ILIKE" or search_literal_op == "NOT PG_ILIKE":
                            search_literal_op = "(NOT ILIKE|NOT PG_ILIKE)"
                        escaped_column = re.escape(str_literal_op["column"])
                        escaped_literal = re.escape(str_literal_op["literal"])
                        search_string = f"{escaped_column} {search_literal_op} '{escaped_literal}'"
                        target_string = f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'"
                        compiled_re = re.compile(search_string, re.IGNORECASE)
                        altered_query = compiled_re.sub(target_string, altered_query)

                        self.logger.debug(
                            f"Operator changed from {str_literal_op['operator']} to {altered_str_literal_op['operator']}"
                        )
                        self.logger.debug(
                            f"Literal changed from {str_literal_op['literal']} to {altered_str_literal_op['literal']}"
                        )
        except Exception as e:
            self.logger.info(f"Error correcting string literals: {e}")
            return query

        self.logger.debug(f"Altered query: {altered_query}")
        return altered_query

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
            all_table_names = table_names

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

                        def matching_prefix_length(s1, s2):
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


heavydb_var: ContextVar[HeavyDB | None] = ContextVar("heavydb_var", default=None)


@asynccontextmanager  # type: ignore
async def heavydb_context(session_id_or_db: str | HeavyDB) -> AsyncGenerator[HeavyDB, None]:
    """
    Async Context which helps to optionally create HeavyDB instance on Setup, set context var and yields it, finally reset
    context var on teardown.

    Example:
        async with heavydb_context(session_id) as db:
            assert heavydb_var.get() == db
    """
    db = session_id_or_db
    if not isinstance(db, HeavyDB):
        db = await HeavyDB.from_session_async(session_id=session_id_or_db)  # type: ignore
    heavydb_var.set(db)
    yield db
    heavydb_var.set(None)


async def get_db(session_id: str) -> HeavyDB:
    """
    Gets heavydb instance from context var if there's any, else create it from session_id.
    Always call this function within a heavydb_context ctx in-order to avoid redundant heavydb conntection calls.
    """
    from heavyiq.logging_utils import get_heavyiq_logger

    db = heavydb_var.get()
    if db:
        return db

    logger = get_heavyiq_logger()
    logger.warning("Establishing HeavyDB connection outside of heavydb_context!")
    return await HeavyDB.from_session_async(session_id=session_id)
