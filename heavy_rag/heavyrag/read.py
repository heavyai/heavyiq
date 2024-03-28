import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from threading import Lock
from typing import Any, Callable, Iterable, Optional

from heavyai.connection import Connection, connect
from llama_index.core.readers.base import BaseReader, Document
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HeavyDBReader(BaseReader):
    """
    Initializes a new instance of the HeavyDBReader.

    This class establishes a connection to HeavyDB using heavydb connector, executes query
    and concatenates each row into Document used by LlamaIndex.

    Attributes:
        uri (Optional[str]): HeavyDB URI (ex: 'heavydb://admin:HyperInteractive@localhost:6274/heavyai?protocol=binary')
        user: str
        password: str
        host: str
        port: int
        dbname: str
        protocol: {'binary', 'http', 'https'}
        sessionid: str
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6274,
        dbname: Optional[str] = None,
        protocol: str = "binary",
        sessionid: Optional[str] = None,
    ) -> None:
        """
        Initializes the HeavyDBReader with optional connection details.

        Args:
            uri: HeavyDB URI. Defaults to None.
            user: HeavyDB user. Defaults to None.
            password: HeavyDB password. Defaults to None.
            host: HeavyDB host. Defaults to None.
            port: HeavyDB port. Defaults to 6274.
            dbname: HeavyDB database name. Defaults to None.
            protocol: protocol to be used. Defaults to "binary".
            sessionid: HeavyDB session id. Defaults to None.
        """

        self.connection = self._connect_with_timeout(
            connect,
            uri=uri,
            user=user,
            password=password,
            host=host,
            port=port,
            dbname=dbname,
            protocol=protocol,
            sessionid=sessionid,
        )
        self._lock = Lock()

    @cached_property
    def database_name(self):
        return self.connection._client.get_session_info(
            self.connection._session
        ).database

    @classmethod
    def _connect_with_timeout(
        cls: type["HeavyDBReader"],
        connect_func: Callable[[], Connection],
        *func_args,
        timeout: float = 5,
        **func_kwargs,
    ) -> Connection:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(connect_func, *func_args, **func_kwargs)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise Exception(f"HeavyDB failed to connect after {timeout} seconds")

    @staticmethod
    def is_destructive_sql(sql: str) -> bool:
        """
        Determines if a provided SQL statement is destructive or causes a modification.

        Args:
            sql (str): The SQL statement to be checked.

        Returns:
            bool: True if the SQL statement is destructive, False if it is non-destructive.
        """
        # Check for destructive SQL statements
        destructive_statements = {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "REPLACE",
            "CREATE",
        }

        # Convert the SQL statement to uppercase and split it by whitespace
        sql_parts = sql.strip().upper().split()

        # Check if the first part of the SQL statement is in the destructive_statements set
        return sql_parts[0] in destructive_statements

    def execute_query(self, query_string: str) -> list[Any]:
        """
        Executes a SQL query and returns the fetched results.

        Args:
            query_string (str): The SQL query to be executed.

        Returns:
            list[Any]: The fetched results from the query.
        """
        try:
            query_string = query_string.strip()
            if self.is_destructive_sql(query_string):
                raise ValueError("Destructive SQL is not allowed")
            with self._lock:
                cursor = self.connection.execute(query_string)
            return cursor.fetchall()
        except Exception as e:
            print("Query execution failed!")
            raise e

    def _create_table_document(self, table_name: str, table_comment: str) -> Document:
        """
        Creates a table document which act as SOURCE node/document for all the child column nodes.
        """
        return Document(
            text=table_comment,
            metadata={"table": table_name, "database": self.database_name},
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent:\n{content}",
        )  # type: ignore

    def read_table_schema(self, table_name: str) -> str | None:
        """
        Grabs the raw table schema from database.
        """
        result, schema = self.execute_query(f"SHOW CREATE TABLE {table_name};"), None
        if result and (first_row := result[0]):
            schema = first_row[0]
        return schema

    @cached_property
    def tables(self) -> list[str]:
        return self.connection.get_tables()

    def read_table_schemas(
        self, exclude_tables: Iterable[str] | None = None
    ) -> list[Document]:
        """
        Read schemas from multiple tables and generate documents based on them.
        If exclude_tables was given, then it generates documents only for the missing tables.
        """
        tables = set(self.tables)
        if exclude_tables:
            tables = tables - set(exclude_tables)

        if not tables:
            return []

        documents = []
        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            future_to_task = {
                executor.submit(self.read_table_schema, table): table
                for table in tables
            }
            with tqdm(desc="Fetching table schemas", total=len(future_to_task)) as pbar:
                for future in as_completed(future_to_task):
                    task_result = future.result()
                    documents.append(
                        self._create_table_document(future_to_task[future], task_result)
                    )
                    pbar.update(1)

        return documents
