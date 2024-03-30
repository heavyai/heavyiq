from functools import lru_cache

import chromadb
from chromadb.api.types import CollectionMetadata
from chromadb.config import Settings as ConfigSettings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavyrag.settings import settings
from heavyrag.transform import TableSchemaSplitter

# metadata relevant to collection types
COLLECTION_TYPE_METADATA = {
    "database": {"type": "database"},
    "documents": {"type": "documents"},
}


def get_vectorstore(
    persist_collection_dir: str,
    collection_name: str,
    metadata: CollectionMetadata | None = None,
) -> ChromaVectorStore:
    """
    Supposed to get the corresponding vectorstore.
    """
    # set up ChromaVectorStore

    chroma_client = chromadb.PersistentClient(
        path=persist_collection_dir,
        settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True),
    )
    chroma_collection = chroma_client.get_or_create_collection(
        collection_name, metadata=metadata
    )
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


def get_hf_embeddings():
    return HuggingFaceEmbedding(model_name=settings.hf_embedding_model)


@lru_cache
def get_or_create_index(collection_name: str | None = None):
    """
    Creates a new index from documents.
    """
    vector_store = get_vectorstore(
        settings.persistant_collection_dir,
        collection_name or settings.collection_name,
        metadata=COLLECTION_TYPE_METADATA["database"],
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        transformations=[
            TableSchemaSplitter(),
            HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
        ],
        insert_batch_size=166,
        embed_model=HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
    )
    return index
