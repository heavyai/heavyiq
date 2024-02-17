from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from heavy_rag.constants import HF_EMBEDDING_MODEL, PERSIST_DIR
from heavy_rag.read import HeavyDBReader
from heavy_rag.transform import TableSchemaSplitter


def reload_index() -> VectorStoreIndex:
    """
    Refetch table schema documents from heavydb, transform each document into multiple nodes, finally load it to chromadb.
    """
    reader = HeavyDBReader(host="10.2.1.33", port=6274, user="admin", password="HyperInteractive", dbname="heavyiq")
    documents = reader.read_table_schemas()

    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
        transformations=[
            TableSchemaSplitter(),
            HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL),
        ],
    )
    index.storage_context.persist(PERSIST_DIR)
    return index
