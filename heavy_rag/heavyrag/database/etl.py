"""
# Extract, Tranform, Load, Retrieve
# 1. Extracts data/table schema from HeavyDB database.
# 2. Transform each table schema to llama-index document, which then again splitted up
#    to form nodes which contain info related to column metadata.
# 3. Load/Ingest the aforementioned nodes into the vectordb index to establish an index where users can pose questions.
# 4. Use the above index as retriever to retieve  the documents relevant to the asked question.
"""

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from heavyrag.common.etl import BaseETL
from heavyrag.database.ingest import sync_index
from heavyrag.database.read import HeavyDBReader
from heavyrag.settings import settings

logger = logging.getLogger(__name__)


class HeavyDBIndexETL(BaseETL):
    """
    Creates a RAG index using HeavyDB table schemas.
    """

    def read(self) -> list[Document]:
        """
        Read table schemas and return it as list of documents.
        """
        return self.reader.load_data()  # type: ignore

    def sync_index(self, collection_name: str) -> VectorStoreIndex:
        """
        Sync index with respect to the changes in the loaded documents
        """
        return sync_index(self.reader)

    def retrieve_nodes(self, index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
        """
        Retrieve nodes from the database index (index built from heavydb table schemas) based on the question.
        """
        # define metadata filters
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="database", operator=FilterOperator.EQ, value=index.vector_store.client.name)
            ]  # type: ignore
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

    def retrieve_tables(self, index: VectorStoreIndex, question: str) -> dict[str, float]:
        """
        Retrieve tables along with it's score from db index.
        """
        nodes_with_score = self.retrieve_nodes(index, question)
        seen: dict[str, float] = {}
        for i in nodes_with_score:
            table = i.node.metadata["table"]
            if table not in seen:
                seen[table] = i.score  # type: ignore
        # returns table_name, score mapping dict
        return seen

    def run(self, question: str) -> list[NodeWithScore]:
        index = self.index(self.reader.database_name)
        nodes_with_scores = self.retrieve_index(index, question)
        return nodes_with_scores


def ask_db_index(sessionid: str, question: str, n: int = 2) -> list[str]:
    """
    Ask a question against the stored index.
    Returns a list of possible tables which are likely to be associated with the asked question.

    Args:
        sessionid (str): HeavyDB session id.
        question (str) : Natural Language Question.
              n (int)  : Number of tables to return.

    Returns:
        A list of matched table names.
    """
    etl = HeavyDBIndexETL(
        reader_cls=HeavyDBReader,
        reader_init_kwargs={"host": settings.heavydb_host, "port": settings.heavydb_port, "sessionid": sessionid},
    )
    index = etl.index(etl.reader.database_name)
    tables_with_score = etl.retrieve_tables(index, question)
    return list(tables_with_score.keys())[:n]
