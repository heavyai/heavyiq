import logging
from typing import Any

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.chroma.base import _to_chroma_filter

logger = logging.getLogger(__name__)


def get_existing_tables_from_collection(collection: Any, dbname: str) -> list[str]:
    """
    Gets the existing tables from ChromaDB collection.
    """
    logger.debug(f'Getting existing tables from the "{dbname}" collection...')
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="database", operator=FilterOperator.EQ, value=dbname),
            MetadataFilter(key="type", operator=FilterOperator.EQ, value="table"),
        ],
        condition=FilterCondition.AND,
    )
    where = _to_chroma_filter(filters)
    query_result = collection.get(where=where, include=["metadatas"])

    tables = list(set([i["table"] for i in query_result["metadatas"]]))
    logger.debug(f"{len(tables)} tables already exists in the vectorstore collection")
    return tables
