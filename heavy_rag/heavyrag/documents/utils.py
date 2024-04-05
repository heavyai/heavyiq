import logging

from chromadb.api.models.Collection import Collection
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma.base import _to_chroma_filter

logger = logging.getLogger(__name__)


def get_existing_doc_ids_from_collection(collection: Collection) -> list[str]:
    """
    Gets the existing document ids from ChromaDB collection.
    """
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="type", operator=FilterOperator.EQ, value="document"),
        ],
    )
    where = _to_chroma_filter(filters)
    query_result = collection.get(where=where, include=["metadatas"])

    doc_ids: list[str] = list(set([i["database_doc_id"] for i in query_result["metadatas"]]))  # type: ignore
    return doc_ids
