import os
from enum import Enum

from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_classic.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import TextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from heavyiq.config import get_config

hf_model = None


def get_or_download_hf_model() -> HuggingFaceEmbeddings:
    """
    Get or download hugging face embedding model.
    """
    from heavyiq.logging_utils import get_heavyiq_logger

    global hf_model
    if hf_model:
        return hf_model

    config, logger = get_config(), get_heavyiq_logger()
    if os.path.exists(config.huggingface_model_cache_folder):
        logger.debug("Initializing HF model embeddings from cache...")
    else:
        logger.debug("Downloading HF model embeddings...")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf_model = HuggingFaceEmbeddings(
        model_name=config.huggingface_embed_model,
        cache_folder=config.huggingface_model_cache_folder,
        multi_process=config.huggingface_embed_documents_parallel,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.debug("Initialized HuggingFace embeddings!")
    return hf_model


def get_vectorstore_index_creator(persist_directory: str) -> VectorstoreIndexCreator:
    """
    Get a VectorstoreIndexCreator instance.
    :param persist_directory: Path to the directory where the vector store index should be persisted.
    :return: A VectorstoreIndexCreator instance.
    """
    return VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": persist_directory},
        embedding=get_or_download_hf_model(),
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
