from enum import Enum

from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import TextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from heavyiq.config import get_config

_embed_model = None


def get_embed_model() -> OpenAIEmbeddings:
    """
    Get embedding model using the custom embedding server from config.
    Uses the same embedding server as the RAG system (rag_embed_server_base).
    """
    from heavyiq.logging_utils import get_heavyiq_logger

    global _embed_model
    if _embed_model:
        return _embed_model

    config, logger = get_config(), get_heavyiq_logger()
    
    if not config.rag_embed_server_base:
        raise ValueError(
            "rag_embed_server_base is not configured. "
            "Please set it in your config file to use the metadata index."
        )
    
    # Use the same embedding server as the RAG system
    logger.debug(f"Initializing embeddings from {config.rag_embed_server_base}...")
    
    _embed_model = OpenAIEmbeddings(
        openai_api_key="nothing",  # Custom server doesn't need real API key
        openai_api_base=config.rag_embed_server_base,
        model=config.rag_embed_model_name or "text-embedding",
    )
    
    logger.debug("Initialized embeddings from custom server!")
    return _embed_model


def get_vectorstore_index_creator(persist_directory: str) -> VectorstoreIndexCreator:
    """
    Get a VectorstoreIndexCreator instance.
    :param persist_directory: Path to the directory where the vector store index should be persisted.
    :return: A VectorstoreIndexCreator instance.
    """
    return VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": persist_directory},
        embedding=get_embed_model(),
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
        search_kwargs = {"k": k}
        return self.vectorstore.as_retriever(search_type=search_type.value, search_kwargs=search_kwargs)
