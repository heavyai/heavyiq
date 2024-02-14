import logging
import re
from typing import Any, List, Optional

from heavydb.connection import connect
from llama_index.core.readers.base import BaseReader, Document

logger = logging.getLogger(__name__)


def is_destructive_sql(sql: str) -> bool:
    """
    Determines if a provided SQL statement is destructive or causes a modification.

    Args:
        sql (str): The SQL statement to be checked.

    Returns:
        bool: True if the SQL statement is destructive, False if it is non-destructive.
    """
    # Check for destructive SQL statements
    destructive_statements = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE"}

    # Convert the SQL statement to uppercase and split it by whitespace
    sql_parts = sql.strip().upper().split()

    # Check if the first part of the SQL statement is in the destructive_statements set
    return sql_parts[0] in destructive_statements


def get_table_names_from_sql(sql: str) -> List[str]:
    """
    Get table names from the input query.
    """
    return re.findall(r"(?i)(?:from|join)\s+(\S+)", sql)


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
        self.connection = connect(
            uri=uri,
            user=user,
            password=password,
            host=host,
            port=port,
            dbname=dbname,
            protocol=protocol,
            sessionid=sessionid,
        )

    def execute_query(self, query_string: str) -> List[Any]:
        """
        Executes a SQL query and returns the fetched results.

        Args:
            query_string (str): The SQL query to be executed.

        Returns:
            List[Any]: The fetched results from the query.
        """
        try:
            query_string = query_string.strip()
            if is_destructive_sql(query_string):
                raise ValueError("Destructive SQL is not allowed")

            cursor = self.connection.execute(query_string)
            return cursor.fetchall()
        finally:
            # Ensure the session is closed after query execution
            self.connection.close()

    def load_data(self, query: str) -> List[Document]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.
        """
        documents = []

        if query is None:
            raise ValueError("A query parameter is necessary to filter the data")

        try:
            result = self.execute_query(query)
            tables = get_table_names_from_sql(query)

            for item in result:
                # create document for each item
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(Document(text=doc_str, metadata={"source": tables}))
        except Exception as e:
            logger.error("An error occurred while loading the data: {}".format(e), exc_info=True)

        return documents
