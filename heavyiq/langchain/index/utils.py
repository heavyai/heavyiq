from enum import Enum

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import BaseRetriever
from langchain.text_splitter import TextSplitter
from langchain.vectorstores.chroma import Chroma

from heavyiq.config import get_config


def get_vectorstore_index_creator(persist_directory: str) -> VectorstoreIndexCreator:
    """
    Get a VectorstoreIndexCreator instance.
    :param persist_directory: Path to the directory where the vector store index should be persisted.
    :return: A VectorstoreIndexCreator instance.
    """
    config = get_config()
    huggingface_model_name = config.huggingface_embed_model
    return VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": persist_directory},
        embedding=HuggingFaceEmbeddings(model_name=huggingface_model_name),
    )


class SearchType(Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"


class HeavyIQIndexWrapper(VectorStoreIndexWrapper):
    text_splitter: TextSplitter
    vectorstore: Chroma

    def as_retriever(
        self,
        search_type: SearchType = SearchType.SIMILARITY,
        k: int = 6,
        fetch_k: int = 20,
    ) -> BaseRetriever:
        """
        Args:
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. Defaults to 6.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to 20.

        Returns:
            Retriever that can be used to search the index.
        """
        search_kwargs = {"k": k, "fetch_k": fetch_k}
        return self.vectorstore.as_retriever(search_type=search_type.value, search_kwargs=search_kwargs)
