import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, Document, NodeWithScore
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from heavyrag.common.etl import BaseETL
from heavyrag.documents.ingest import sync_doc_index
from heavyrag.documents.read import DocumentDatabaseReader, read_documents

logger = logging.getLogger(__name__)


class DocumentIndexETL(BaseETL):
    """
    Creates a RAG index using the uploaded documents.
    """

    collection_name: str = "documents"

    def read(self) -> list[Document]:
        """
        Read table schemas and return it as list of documents.
        """
        return read_documents(self.reader)  # type: ignore

    def sync_index(self, collection_name: str) -> VectorStoreIndex:
        """
        Sync index with respect to the changes in the loaded documents
        """
        return sync_doc_index(self.reader, collection_name)

    def retrieve_nodes(self, index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
        """
        Retrieve nodes from the database index (index built from heavydb table schemas) based on the question.
        """
        # define metadata filters
        filters = MetadataFilters(
            filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="document")]  # type: ignore
        )
        retriever = index.as_retriever(
            similarity_top_k=10,
            filters=filters,
        )
        nodes: list[NodeWithScore] = retriever.retrieve(question)
        return nodes

    def retrieve_index(self, index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
        """
        Retrieves relevant nodes from the index.
        """
        return self.retrieve_nodes(index, question)

    def run(self, question: str) -> list[NodeWithScore]:
        index = self.index(self.collection_name)
        nodes_with_scores = self.retrieve_index(index, question)
        return nodes_with_scores


def ask_doc_index(question: str, query: str | None = None, n: int = 2, **dbargs) -> list[BaseNode]:
    """
    Ask a question against the stored document index.

    Args:
        question (str): NL question
        query (str|None): Optional database retrieve query for revieving documents from SQLite database.

    Kwargs:
        dbargs : kwargs used to instantiate DocumentDatabaseReader

    Returns:
        list[str]:
    """

    etl = DocumentIndexETL(
        reader_cls=DocumentDatabaseReader,
        reader_init_kwargs=dbargs,
    )
    nodes_with_score = etl.run(question)
    return [i.node for i in nodes_with_score[:n]]
