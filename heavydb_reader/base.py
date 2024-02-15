import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Iterable, List, Optional

from heavyai.connection import connect
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.readers.base import BaseReader, Document
from llama_index.core.schema import MetadataMode, TextNode
from tqdm import tqdm

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


def get_table_name_from_sql(sql: str) -> List[str]:
    """
    Get table names from the input query.
    """
    return re.findall(r"(?i)from\s+(\S+)", sql)


# compiled regex used to grab table and column comments from table create query response.
comment_regex = rgx = re.compile(
    r"(?mi)create table (?P<table_name>\S+)\s*(?:/\*\s*(?P<table_comment>.*?)\s*\*\/)?\s*\($|^\s*(\w\S*).*?(?:\/\*\s*(.*?)\s*\*\/)?(?:,|\);)?$"
)


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
        self.dbname = dbname
        self._lock = Lock()

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
            with self._lock:
                cursor = self.connection.execute(query_string)
            return cursor.fetchall()
        except Exception as e:
            print("Query execution failed!")
            raise e

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
            table = get_table_name_from_sql(query)

            for item in result:
                # create document for each item
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(Document(text=doc_str, metadata={"table": table, "database": self.dbname}))
        except Exception as e:
            logger.error("An error occurred while loading the data: {}".format(e), exc_info=True)

        return documents

    def get_table_comments(self, table_name: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Get table comment and all column comments.

        Ex:
        <table_comment>, [("column_a", "comment of column a")]
        """
        result, table_comment = self.execute_query(f"SHOW CREATE TABLE {table_name};"), ""
        columns = []
        if result and (first_row := result[0]):
            raw_table_schema = first_row[0]
            matches = comment_regex.findall(raw_table_schema)
            for found in matches:
                _, table_comment, column, column_comment = found
                if column:
                    columns.append((column, column_comment))

        return table_comment, columns

    def load_table_schema_thrift(self, table_name: str) -> List[TextNode]:
        """
        Grab table schema from database through thrift endpoint, form a parent node for table comment and the relevant child nodes for the columns.
        """
        # Get table comment and each column comment from thrift endpoint
        table_details = self.connection._client.get_table_details(self.connection._session, table_name)
        table_comment = getattr(table_details.refresh_info, "comment", f"{table_name}: table comment")
        table_document = self._create_table_document(table_name=table_name, table_comment=table_comment)

        splits = []
        for col in table_details.row_desc:
            column_name = col.col_name
            column_comment = getattr(col, "comment", f"{table_name}.{column_name}: column comment")
            splits.append(column_comment)

        # create table_comment as separate Document and the relevant columns as
        # TextNode's having the parent as table_comment
        nodes = build_nodes_from_splits(splits, table_document)
        return nodes

    def _create_table_document(self, table_name: str, table_comment: str) -> Document:
        """
        Creates a table document which act as SOURCE node/document for all the child column nodes.
        """
        return Document(
            text=table_comment,
            metadata={"table": table_name, "database": self.dbname},
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent:\n{content}",
        )

    def load_table_schema_query(self, table_name: str) -> List[TextNode]:
        """
        Load table schema through SQL query.
        This function returns a list of TextNodes, in-order to create a new index.
        """
        logger.info(f"Deriving nodes for {table_name}...")
        table_comment, columns = self.get_table_comments(table_name=table_name)
        table_document, splits = self._create_table_document(table_name=table_name, table_comment=table_comment), []

        column_text_template = "{table_name}.{column_name}: {column_comment}"
        for col_name, col_comment in columns:
            splits.append(
                column_text_template.format(table_name=table_name, column_name=col_name, column_comment=col_comment)
            )

        nodes = build_nodes_from_splits(splits, table_document)
        logger.info(f"Successfully found nodes for {table_name}...")
        return nodes

    def load_tables(self, tables: Optional[Iterable[str]] = None) -> List[TextNode]:
        """
        Generate nodes for all the tables that exist within the database where the current HeavyDB connection was established.
        """
        if not tables:
            tables = self.connection.get_tables()

        nodes = []
        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            future_to_task = {executor.submit(self.load_table_schema_query, table): table for table in tables}
            with tqdm(total=len(future_to_task)) as pbar:
                for future in as_completed(future_to_task):
                    task_result = future.result()
                    nodes.extend(task_result)
                    pbar.update(1)

        return nodes
