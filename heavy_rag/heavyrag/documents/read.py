"""
Read raw documents from SQlite database table using DBReader 
"""

from typing import List

from llama_index.core.schema import Document
from llama_index.readers.database import DatabaseReader
from sqlalchemy import text


class DocumentDatabaseReader(DatabaseReader):
    """
    DatbaseReader responsible for reading documents from an SQlite database table.
    """

    def load_data(self, query: str) -> List[Document]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): Query parameter to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.
        """
        documents = []
        with self.sql_database.engine.connect() as connection:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = connection.execute(text(query))

            for item in result.fetchall():
                # fetch each item
                doc_str = ", ".join([f"{col}: {entry}" for col, entry in zip(result.keys(), item)])
                documents.append(Document(text=doc_str))
        return documents


def read_documents(reader: DocumentDatabaseReader, query: str | None = None) -> list[Document]:
    """
    Read documents from an SQLite Database engine or URI or etc.
    """
    query = (
        query
        or f"""
        SELECT
            id, content AS text
        FROM documents.documents;
    """
    )
    return reader.load_data(query=query)
