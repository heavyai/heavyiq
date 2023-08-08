from __future__ import annotations
import re
import multiprocessing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Any, Iterable, TYPE_CHECKING, Callable

from heavyai import connect, Connection
from heavyiq.config import get_config
from heavyiq.utils import strip_sql_comments, is_destructive_sql, rate_sql_complexity, LRUCache

if TYPE_CHECKING:
    from heavydb._parsers import ColumnDetails


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


class HeavyDB:
    """A heavydb database connection."""

    _manager = multiprocessing.Manager()
    # single manager process being shared with all the caches
    top_k_cache = LRUCache[str, str](manager=_manager)
    sample_rows_cache = LRUCache[str, str](manager=_manager)
    table_schema_cache = LRUCache[str, str](manager=_manager)

    def __init__(
        self,
        conn: Connection,
        ignore_tables: Optional[list[str]] = None,
        include_tables: Optional[list[str]] = None,
        sample_rows_in_table_info: int = 2,
        custom_table_info: Optional[dict[str, str]] = None,
    ):
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._conn = conn
        self.lock = Lock()

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

    def __del__(self):
        try:
            self._conn.close()
        except Exception as e:
            print(f"Error: {e}")

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
    def from_uri(cls: type[HeavyDB], database_uri: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a database URI."""
        conn = cls._connect_with_timeout(lambda: connect(database_uri), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_env(cls: type[HeavyDB], **kwargs: Any) -> HeavyDB:
        """Create a database connection from environment variables."""
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password or not config.heavydb_dbname:
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        def connect_func() -> Connection:
            return connect(
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=config.heavydb_dbname,
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

    def get_usable_table_names(self) -> Iterable[str]:
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

    @property
    def table_info(self) -> str:
        """Information about all usable tables in the database."""
        return self.get_table_info()

    def get_table_columns(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        with self.lock:
            return self._conn.get_table_details(table)

    def get_table_schema(self, table: str) -> str:
        cached_value = self.table_schema_cache.get(table)
        if cached_value is not None:
            return cached_value
        """Get the schema of a table."""
        create_command = f"SHOW CREATE TABLE {table};"
        with self.lock:
            cursor = self._conn.execute(create_command)
        table_schema = cursor.fetchone()[0]  # type: ignore
        table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)  # type: ignore
        table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
        table_schema = re.sub(r"\n", "", table_schema)
        if "WITH (" in table_schema:
            table_schema = table_schema[: table_schema.index("WITH (")]
        self.table_schema_cache.put(table, table_schema)
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

    def get_column_top_k(self, table: str, column: str, k: int = 5) -> Optional[list[str]]:
        """Get the top k values for a column."""
        top_k_statement = f"SELECT {column}, COUNT(*) as cnt FROM {table} WHERE {column} is not null GROUP BY {column} ORDER BY cnt DESC LIMIT {k};"
        with self.lock:
            cursor = self._conn.execute(top_k_statement)
        top_k_res: list[str] = [str(v[0]) for v in cursor.fetchall()]
        if not any([v for v in top_k_res if v.startswith("MULTIPOLYGON")]):
            return top_k_res
        return None

    def get_sample_rows(self, table_name: str) -> str:
        cached_value = self.sample_rows_cache.get(table_name)
        if cached_value is not None:
            return cached_value
        # build the select command
        command = f"SELECT * FROM {table_name} LIMIT {self._sample_rows_in_table_info}"

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

        self.sample_rows_cache.put(table_name, res)

        return res

    def get_top_k(self, table_name: str) -> str:
        cached_value = self.top_k_cache.get(table_name)
        if cached_value is not None:
            return cached_value
        text_columns = [
            c.name
            for c in self.get_table_columns(table_name)
            if c.type == "STR" and c.encoding == "DICT" and c.is_array is False
        ]
        top_k_strings = "Sample values for text columns (comma-separated):\n"
        for col in text_columns:
            top_k_res = self.get_column_top_k(table_name, col)
            if top_k_res:
                top_k_strings += f"{col}: {', '.join(top_k_res)}\n"
        self.top_k_cache.put(table_name, top_k_strings)
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
            all_table_names = table_names

        tables = []
        for table in all_table_names:
            tables.append(self.get_single_table_info(table, **kwargs))

        final_str = "\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
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
        return str(result)

    def explain(self, command: str) -> str:
        command = strip_sql_comments(command)
        if is_destructive_sql(command):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(f"EXPLAIN plan {command}")
        result: tuple[str] = cursor.fetchone()  # type: ignore
        return str(result[0])

    def complexity(self, command: str) -> int:
        plan = self.explain(command)
        return rate_sql_complexity(plan)

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
            return self.run(command, fetch)
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}"
