import chromadb
from chromadb.config import Settings as ConfigSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rag.read import HeavyDBReader
from rag.transform import TableSchemaSplitter
from settings import Settings


def get_vectorstore(persist_collection_dir: str, collection_name: str = "langchain") -> ChromaVectorStore:
    """
    Supposed to get the corresponding vectorstore.
    """
    # set up ChromaVectorStore

    chroma_client = chromadb.PersistentClient(
        path=persist_collection_dir, settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True)
    )
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


def reload_index(
    reader: HeavyDBReader,
    settings: Settings | None = None,
    exclude_tables: list[str] | None = None,
) -> VectorStoreIndex:
    """
    Refetch table schema documents from heavydb, transform each document into multiple nodes, finally load it to chromadb.
    """
    settings = settings if settings else Settings()
    storage_context = StorageContext.from_defaults(
        vector_store=get_vectorstore(settings.persistant_collection_dir, collection_name=settings.collection_name)
    )

    documents = reader.read_table_schemas(exclude_tables=exclude_tables)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
        transformations=[
            TableSchemaSplitter(),
            HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
        ],
        storage_context=storage_context,
        insert_batch_size=166,
    )
    # stores the index mapping documents locally
    index.storage_context.persist(settings.persistant_storage_dir)
    return index
