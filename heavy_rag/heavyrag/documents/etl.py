import logging
from functools import cached_property

from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, Document, NodeWithScore
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from heavyrag.common.etl import BaseETL
from heavyrag.common.synthesizer import response_synthesizer
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
        retriever = index.as_retriever(
            similarity_top_k=2,
            filters=self.default_filters,
        )
        nodes: list[NodeWithScore] = retriever.retrieve(question)
        return nodes

    @cached_property
    def default_filters(self):
        """
        Default retiever filters.
        """
        return MetadataFilters(
            filters=[MetadataFilter(key="type", operator=FilterOperator.EQ, value="document")]  # type: ignore
        )

    def retrieve_index(self, index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
        """
        Retrieves relevant nodes from the index.
        """
        return self.retrieve_nodes(index, question)

    @cached_property
    def idx(self):
        """
        Helps to return the index.
        """
        return self.index(self.collection_name)

    def run(self, question: str) -> RESPONSE_TYPE:
        """
        Retrieve nodes and then generate an answer from its content by building it as a prompt context, which is then passed to the language model.
        """
        # configure retriever
        retriever = VectorIndexRetriever(index=self.idx, similarity_top_k=2, filters=self.default_filters)
        # configure response synthesizer
        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.35)],
        )
        # query
        response = query_engine.query(question)
        return response


def ask_doc_index(question: str, n: int = 2, **dbargs) -> tuple[str, dict]:
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
    response = etl.run(question)
    return response.response, response.metadata  # type: ignore
