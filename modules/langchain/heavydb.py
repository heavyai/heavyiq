from __future__ import annotations
from functools import lru_cache
import os
from threading import Lock
from typing import Optional, Any, Iterable, TYPE_CHECKING

from heavyai import connect
from dotenv import load_dotenv

if TYPE_CHECKING:
    from heavyai import Connection
    from heavydb._parsers import ColumnDetails

load_dotenv()

heavydb_host = os.getenv("HEAVYDB_HOST")
heavydb_port = os.getenv("HEAVYDB_PORT")
heavydb_dbname = os.getenv("HEAVYDB_DBNAME")
heavydb_username = os.getenv("HEAVYDB_USERNAME")
heavydb_password = os.getenv("HEAVYDB_PASSWORD")


class HeavyDB:
    """A heavydb database connection."""

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
                raise ValueError(f"ignore_tables {missing_tables} not found in database")

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
    def from_uri(cls: type[HeavyDB], database_uri: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a database URI."""
        return cls(connect(database_uri), **kwargs)

    @classmethod
    def from_env(cls: type[HeavyDB], **kwargs: Any) -> HeavyDB:
        """Create a database connection from environment variables."""
        conn: Connection = connect(
            user=heavydb_username,
            password=heavydb_password,
            host=heavydb_host,
            port=heavydb_port,
            dbname=heavydb_dbname,
        )
        return cls(conn, **kwargs)

    @classmethod
    def from_session(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a session id."""
        conn: Connection = connect(sessionid=session_id, host=heavydb_host, port=heavydb_port)
        return cls(conn, **kwargs)

    @classmethod
    def from_creds(cls: type[HeavyDB], username: str, password: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from username and password."""
        conn: Connection = connect(
            user=username,
            password=password,
            host=heavydb_host,
            port=heavydb_port,
            dbname=heavydb_dbname,
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

    @property
    def table_info(self) -> str:
        """Information about all usable tables in the database."""
        return self.get_table_info()

    @lru_cache
    def get_table_columns(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        return self._conn.get_table_details(table)

    @lru_cache
    def get_table_schema(self, table: str) -> str:
        """Get the schema of a table."""
        create_command = f"SHOW CREATE TABLE {table};"
        cursor = self._conn.execute(create_command)
        return cursor.fetchone()[0]

    def validate_query(self, query: str) -> list:
        """Validate a query."""
        if "*/" in query:
            # remove block comment from begining of query
            query = query.split("*/", 1)[1]
        return self._conn._client.sql_validate(self._conn._session, query)

    def validate_session(self, session_id: str) -> bool:
        return self._conn._client.get_session_info(session_id)

    @lru_cache
    def get_column_top_k(self, table: str, column: str, k: int = 5) -> Optional[list[str]]:
        """Get the top k values for a column."""
        top_k_statement = f"SELECT {column}, COUNT(*) as cnt FROM {table} WHERE {column} is not null GROUP BY {column} ORDER BY cnt DESC LIMIT {k};"
        cursor = self._conn.execute(top_k_statement)
        top_k_res: list[str] = [v[0] for v in cursor.fetchall()]
        if not any([v for v in top_k_res if v.startswith("MULTIPOLYGON")]):
            return top_k_res
        return None

    @lru_cache
    def get_sample_rows(self, table_name: str) -> str:
        # build the select command
        command = f"SELECT * FROM {table_name} LIMIT {self._sample_rows_in_table_info}"

        # save the columns in string format
        columns_str = ",".join([col.name for col in self.get_table_columns(table_name)])

        # get the sample rows
        sample_rows = self._conn.execute(command)
        # shorten values in the sample rows
        sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

        # save the sample rows in string format
        sample_rows_str = "\n".join([",".join(row) for row in sample_rows])

        return f"{self._sample_rows_in_table_info} rows from {table_name} table:\n{columns_str}\n{sample_rows_str}"

    @lru_cache
    def get_top_k(self, table_name: str) -> str:
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
        if "*/" in command:
            # remove block comment at begining of command
            command = command.split("*/", 1)[1]
        cursor = self._conn.execute(command)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")
        return str(result)

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
