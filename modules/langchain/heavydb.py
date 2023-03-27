from __future__ import annotations
import os
from typing import Optional, Any, Iterable, TYPE_CHECKING

from heavyai import connect

if TYPE_CHECKING:
    from heavyai import Connection
    from heavydb._parsers import ColumnDetails

heavydb_host = os.getenv("HEAVYDB_HOST")
heavydb_port = os.getenv("HEAVYDB_PORT")
heavydb_dbname = os.getenv("HEAVYDB_DBNAME")
heavydb_username = os.getenv("HEAVYDB_USERNAME")
heavydb_password = os.getenv("HEAVYDB_PASSWORD")
heavydb_protocol = os.getenv("HEAVYDB_PROTOCOL")


class HeavyDB:
    """A heavydb database connection."""

    def __init__(
        self,
        conn: Connection,
        ignore_tables: Optional[list[str]] = None,
        include_tables: Optional[list[str]] = None,
        sample_rows_in_table_info: int = 3,
        custom_table_info: Optional[dict[str, str]] = None,
    ):
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._conn = conn

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
            protocol=heavydb_protocol,
        )
        return cls(conn, **kwargs)

    @classmethod
    def from_session(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a session id."""
        conn: Connection = connect(
            sessionid=session_id, host=heavydb_host, port=heavydb_port, dbname=heavydb_dbname, protocol=heavydb_protocol
        )
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
            protocol=heavydb_protocol,
        )
        return cls(conn, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return "ANSI SQL"

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_details(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        return self._conn.get_table_details(table)

    def get_table_schema(self, table: str) -> str:
        """Get the schema of a table."""
        table_details: list[ColumnDetails] = self.get_table_details(table)
        res = f"CREATE TABLE {table} (\n"
        res += ",\n".join([f"{col.name} {col.type}" for col in table_details]) + "\n"
        res += ");\n"
        return res

    def get_table_info(self, table_names: Optional[list[str]] = None) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        tables = []
        for table in all_table_names:
            if self._custom_table_info and table in self._custom_table_info:
                tables.append(self._custom_table_info[table])
                continue

            table_schema = self.get_table_schema(table)

            if self._sample_rows_in_table_info:
                # build the select command
                command = f"SELECT * FROM {table} LIMIT {self._sample_rows_in_table_info}"

                # save the columns in string format
                columns_str = "\t".join([col.name for col in self.get_table_details(table)])

                # get the sample rows
                sample_rows = self._conn.execute(command)
                # shorten values in the sample rows
                sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

                # save the sample rows in string format
                sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

                table_info = (
                    f"{table_schema.rstrip()}\n"
                    f"/*\n"
                    f"{self._sample_rows_in_table_info} rows from {table} table:\n"
                    f"{columns_str}\n"
                    f"{sample_rows_str}\n"
                    f"*/\n"
                )
                # build final info for table
                tables.append(table_info)
            else:
                tables.append(table_schema)

        final_str = "\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        cursor = self._conn.execute(command)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")
        print(str(result))
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
