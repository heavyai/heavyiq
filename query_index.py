import os

import chromadb
from chromadb.config import Settings as ConfigSettings
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavy_rag.constants import HF_EMBEDDING_MODEL, PERSIST_DIR
from heavy_rag.ingest import reload_index

os.environ["ANONYMIZED_TELEMETRY"] = "false"


# define a global embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)


chroma_client = chromadb.PersistentClient(
    path="./chroma-client", settings=ConfigSettings(anonymized_telemetry=False, is_persistent=True)
)
chroma_collection = chroma_client.get_or_create_collection("langchain")
# set up ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


def get_or_create_index() -> VectorStoreIndex:
    if not os.path.exists(PERSIST_DIR):
        print("Creating new index...")
        index = reload_index()
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))

    return index


index = get_or_create_index()
# # configure retriever
retriever = index.as_retriever(
    similarity_top_k=10,
    filters=MetadataFilters(filters=[MetadataFilter(key="database", operator=FilterOperator.EQ, value="heavyiq")]),
)
out: list[NodeWithScore] = retriever.retrieve("How many states exists?")
# print(out)
for i in out:
    print(i.node.metadata["table"], i.score)
