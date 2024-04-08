"""
Read raw documents from SQlite database table using DBReader 
"""

from io import BytesIO
from typing import List

from llama_index.core.schema import Document
from llama_index.readers.database import DatabaseReader
from sqlalchemy import text

from heavyrag.common.overrides import CustomPDFReader
from heavyrag.documents.enums import DocType


class DocumentDatabaseReader(DatabaseReader):
    """
    DatbaseReader responsible for reading documents from an SQlite database table.
    """

    def load_data(self, query: str) -> List[Document]:
        """
        Query and load data from the Database, returning a list of Documents.
        Table which holds the documents must posses the following fields,
           - id
           - name
           - content
           - document_type

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
                column_data_mapping = dict(zip(result.keys(), item))
                doc_type = column_data_mapping["document_type"]
                file_name = column_data_mapping["name"]
                doc_id = column_data_mapping["id"]
                content = column_data_mapping["content"]

                doc_metadata = {"database_doc_id": doc_id, "type": "document"}

                if doc_type == DocType.PDF.value:
                    doc_metadata["doc_type"] = DocType.PDF.value
                    splitted_documents = CustomPDFReader().load_data_from_file_object(
                        BytesIO(content), file_name=file_name, extra_info=doc_metadata
                    )
                    documents.extend(splitted_documents)
                elif doc_type == DocType.TXT.value:
                    doc_metadata["doc_type"] = DocType.TXT.value
                    doc_metadata["file_name"] = file_name
                    documents.append(Document(text=content, metadata=doc_metadata))  # type: ignore
                else:
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
            *
        FROM documents;
    """
    )
    return reader.load_data(query=query)
