from functools import lru_cache

import chromadb
from chromadb.api.types import CollectionMetadata
from chromadb.config import Settings as ConfigSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavyrag.settings import settings


def get_hf_embeddings():
    return HuggingFaceEmbedding(model_name=settings.hf_embedding_model)


# metadata relevant to collection types
COLLECTION_TYPE_METADATA = {
    "database": {"type": "database"},
    "documents": {"type": "documents"},
}


def get_chromadb_client(persist_dir: str):
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True),
    )


def get_vectorstore(
    persist_collection_dir: str,
    collection_name: str,
    metadata: CollectionMetadata | None = None,
) -> ChromaVectorStore:
    """
    Supposed to get the corresponding vectorstore.
    """
    # set up ChromaVectorStore

    chroma_client = get_chromadb_client(persist_collection_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name, metadata=metadata)
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


def delete_collection(persist_collection_dir: str, collection_name: str):
    """
    Helps to delete a collection.
    """
    chroma_client = get_chromadb_client(persist_collection_dir)

    chroma_client.delete_collection(collection_name)
