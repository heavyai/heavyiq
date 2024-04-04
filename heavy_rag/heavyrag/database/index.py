from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from heavyrag.common import COLLECTION_TYPE_METADATA, get_vectorstore
from heavyrag.database.transform import TableSchemaSplitter
from heavyrag.settings import settings


@lru_cache
def get_or_create_heavydb_index(collection_name: str):
    """
    Get or creates new index from heavydb table schema documents.
    """
    vector_store = get_vectorstore(
        settings.persistant_collection_dir,
        collection_name,
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
