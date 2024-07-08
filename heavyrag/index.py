# Module for creating various indexes

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavyiq.config import get_config
from heavyiq.logging_utils import get_heavyiq_logger
from heavyrag.embed import LlamaIndexEmbeddingAdapter, get_chroma_client, get_embed_model

CONFIG = get_config()
logger = get_heavyiq_logger()
DOCUMENTS_INDEX_ID = "documents"
HEAVYDB_INDEX_ID = "heavydb"
DOCUMENTS_INDEX_NODE = IndexNode(index_id=DOCUMENTS_INDEX_ID, text="Document Index")
HEAVYDB_INDEX_NODE = IndexNode(index_id=HEAVYDB_INDEX_ID, text="HeavyDB Index")

embed_model = get_embed_model()
chroma_client = get_chroma_client()


def get_vectorstore(
    collection_name: str, metadata: dict | None = None, create_collection_if_not_exists: bool = True
) -> ChromaVectorStore | None:
    """
    Get or Create a VectorStore.
    """
    embedding_function = LlamaIndexEmbeddingAdapter(embed_model)
    if create_collection_if_not_exists:
        chroma_collection = chroma_client.get_or_create_collection(
            collection_name, metadata=metadata, embedding_function=embedding_function
        )
    else:
        try:
            chroma_collection = chroma_client.get_collection(collection_name, embedding_function=embedding_function)
        except ValueError:
            return None
    return ChromaVectorStore.from_collection(chroma_collection)


def get_or_create_index(
    collection_name: str, collection_metadata: dict | None = None, use_async: bool = False
) -> VectorStoreIndex:
    """
    Get or create new index if not exists.
    """
    vector_store = get_vectorstore(collection_name, metadata=collection_metadata, create_collection_if_not_exists=True)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=use_async,
        insert_batch_size=166,
        embed_model=embed_model,
    )
    return index


def get_index(collection_name: str) -> VectorStoreIndex | None:
    """
    Get index only by collection name.
    """
    vector_store = get_vectorstore(collection_name, create_collection_if_not_exists=False)
    if not vector_store:
        return None
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index


async def acreate_index_and_insert_nodes(
    nodes: list[BaseNode],
    collection_name: str,
    collection_metadata: dict | None = None,
    use_async: bool = True,
) -> VectorStoreIndex:
    """
    Creates a new VectorStoreIndex from the passed nodes.
    """
    vector_store = get_vectorstore(collection_name, collection_metadata, create_collection_if_not_exists=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=166,
        embed_model=embed_model,
        use_async=use_async,
    )
    index.storage_context.persist(persist_dir=CONFIG.rag_storage_persist_dir)
    return index
