# Module for creating various indexes
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from heavyiq.config import get_config
from heavyiq.logging_utils import get_heavyiq_logger
from heavyrag.embed import get_embed_model
from heavyrag.vector_stores.base import VectorStoreType
from heavyrag.vector_stores.chroma import ChromaIQVectorStore
from heavyrag.vector_stores.faiss import FaissIQVectorStore

CONFIG = get_config()
logger = get_heavyiq_logger()
DOCUMENTS_INDEX_ID = "documents"
HEAVYDB_INDEX_ID = "heavydb"
DOCUMENTS_INDEX_NODE = IndexNode(index_id=DOCUMENTS_INDEX_ID, text="Document Index")
HEAVYDB_INDEX_NODE = IndexNode(index_id=HEAVYDB_INDEX_ID, text="HeavyDB Index")

embed_model = get_embed_model()
vectordb_persist_dir: str | None = None


def get_vectorstore(
    collection_name: str | None = None, metadata: dict | None = None, create_collection_if_not_exists: bool = True
) -> BasePydanticVectorStore | None:
    """
    Get or Create a VectorStore.
    """
    global vectordb_persist_dir
    vector_type = VectorStoreType.from_name(CONFIG.rag_vectordb_type)
    if vector_type == VectorStoreType.CHROMA:
        assert collection_name, "ChromaVectorStore expects a collection name to instantiate"
        vectordb_persist_dir = CONFIG.rag_chromadb_persist_dir
        return ChromaIQVectorStore(
            collection_name=collection_name,
            metadata=metadata,
            create_collection_if_not_exists=create_collection_if_not_exists,
        )
    if vector_type == VectorStoreType.FAISS:

        vectordb_persist_dir = CONFIG.rag_faiss_persist_dir
        return FaissIQVectorStore(persist_dir=CONFIG.rag_faiss_persist_dir)
    raise ValueError("Invalid vectorstore.")


def get_or_create_index(
    collection_name: str | None = None, collection_metadata: dict | None = None, use_async: bool = False
) -> BaseIndex:
    """
    Get or create new index if not exists.
    """
    vector_store = get_vectorstore(collection_name, metadata=collection_metadata, create_collection_if_not_exists=True)
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=CONFIG.rag_faiss_persist_dir
        )
        index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
    except FileNotFoundError:
        # index not exists so create an empty index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=embed_model)
        index.storage_context.persist(persist_dir=vectordb_persist_dir)

    return index


def get_index(collection_name: str | None = None) -> BaseIndex | None:
    """
    Get index only by collection name.
    """
    vector_store = get_vectorstore(collection_name, create_collection_if_not_exists=False)
    if not vector_store:
        return None
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=vectordb_persist_dir)
    index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
    return index


async def acreate_index_and_insert_nodes(
    nodes: list[BaseNode],
    collection_name: str | None = None,
    collection_metadata: dict | None = None,
    use_async: bool = True,
) -> VectorStoreIndex:
    """
    Creates a new VectorStoreIndex from the passed nodes.
    """
    vector_store = get_vectorstore(collection_name, collection_metadata, create_collection_if_not_exists=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=vectordb_persist_dir)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
        insert_batch_size=166,
        embed_model=embed_model,
        use_async=use_async,
    )
    index.storage_context.persist(persist_dir=vectordb_persist_dir)
    return index
