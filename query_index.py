import os
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma.base import ChromaVectorStore, _to_chroma_filter

from heavy_rag.constants import HF_EMBEDDING_MODEL
from heavy_rag.constants import PERSIST_STORAGE_DIR as PERSIST_DIR
from heavy_rag.ingest import get_vectorstore, reload_index
from heavy_rag.read import HeavyDBReader

os.environ["ANONYMIZED_TELEMETRY"] = "false"


# define a global embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL)


def get_existing_tables_from_collection(collection: Any, dbname: str) -> list[str]:
    """
    Gets the existing tables from ChromaDB collection.
    """
    query = VectorStoreQuery(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(key="database", operator=FilterOperator.EQ, value=dbname),
                MetadataFilter(key="type", operator=FilterOperator.EQ, value="table"),
            ],
            condition=FilterCondition.AND,
        )
    )
    where = _to_chroma_filter(query.filters)
    query_result = collection.get(where=where, include=["metadatas"])
    return list(set([i["table"] for i in query_result["metadatas"]]))


def get_or_create_index() -> VectorStoreIndex:
    """
    Get or create VectorStore Index.
    if not exists, create a new one by querying the heavydb for all the table schemas and then load it to the index.
    if exists, then load only the missing tables into the index.
    """
    reader = HeavyDBReader(host="10.2.1.33", port=6274, user="admin", password="HyperInteractive", dbname="heavyiq")
    if not os.path.exists(PERSIST_DIR):
        print("Creating new index...")
        index = reload_index(reader)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=PERSIST_DIR, vector_store=get_vectorstore())
        )
        # grab the chromadb collection and then search for existing tables
        existing_tables = get_existing_tables_from_collection(index._vector_store.client, "heavyiq")
        if existing_tables and (sorted(reader.tables) != sorted(existing_tables)):
            # reload index in-case of missing tables
            index = reload_index(reader, exclude_tables=existing_tables)

    return index


index = get_or_create_index()
# # configure retriever
retriever = index.as_retriever(
    similarity_top_k=10,
    filters=MetadataFilters(filters=[MetadataFilter(key="database", operator=FilterOperator.EQ, value="heavyiq")]),
)
out: list[NodeWithScore] = retriever.retrieve("Which state has the highest number of Points of Interest (POIs)?")
# print(out)
for i in out:
    print(i.node.metadata["table"], i.node.metadata["type"], i.score)
