import chromadb
from llama_index.core import VectorStoreIndex
from chromadb.config import Settings as ConfigSettings
from heavyrag.transform import TableSchemaSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from heavyrag.settings import settings
from functools import lru_cache


@lru_cache
def get_vectorstore(
    persist_collection_dir: str, collection_name: str
) -> ChromaVectorStore:
    """
    Supposed to get the corresponding vectorstore.
    """
    # set up ChromaVectorStore

    chroma_client = chromadb.PersistentClient(
        path=persist_collection_dir,
        settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True),
    )
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


@lru_cache
def get_hf_embeddings():
    return HuggingFaceEmbedding(model_name=settings.hf_embedding_model)


@lru_cache
def get_or_create_index():
    """
    Creates a new index from documents.
    """
    vector_store = get_vectorstore(
        settings.persistant_collection_dir, settings.collection_name
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
