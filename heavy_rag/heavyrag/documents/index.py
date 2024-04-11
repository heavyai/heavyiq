from functools import lru_cache

from heavyrag.common import COLLECTION_TYPE_METADATA, get_vectorstore
from heavyrag.settings import settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@lru_cache
def get_or_create_document_index(collection_name: str, use_async: bool = False) -> VectorStoreIndex:
    """
    Get or creates new index from user uploaded documents.
    """
    vector_store = get_vectorstore(
        settings.persistant_collection_dir,
        collection_name,
        metadata=COLLECTION_TYPE_METADATA["documents"],
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=use_async,
        insert_batch_size=166,
        embed_model=HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
    )
    return index
