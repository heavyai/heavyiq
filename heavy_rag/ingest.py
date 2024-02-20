import chromadb
from chromadb.config import Settings as ConfigSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavy_rag.constants import HF_EMBEDDING_MODEL, PERSIST_COLLECTION_DIR, PERSIST_STORAGE_DIR
from heavy_rag.read import HeavyDBReader
from heavy_rag.transform import TableSchemaSplitter


def get_vectorstore(collection_name: str = "langchain") -> ChromaVectorStore:
    """
    Supposed to get the corresponding vectorstore.
    """
    # set up ChromaVectorStore

    chroma_client = chromadb.PersistentClient(
        path=PERSIST_COLLECTION_DIR, settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True)
    )
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


def reload_index(reader: HeavyDBReader, exclude_tables: list[str] | None = None) -> VectorStoreIndex:
    """
    Refetch table schema documents from heavydb, transform each document into multiple nodes, finally load it to chromadb.
    """
    storage_context = StorageContext.from_defaults(vector_store=get_vectorstore())

    documents = reader.read_table_schemas(exclude_tables=exclude_tables)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
        transformations=[
            TableSchemaSplitter(),
            HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL),
        ],
        storage_context=storage_context,
        insert_batch_size=166,
    )
    # stores the index mapping documents locally
    index.storage_context.persist(PERSIST_STORAGE_DIR)
    return index
