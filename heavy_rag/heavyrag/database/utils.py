import logging

from chromadb.api.models.Collection import Collection
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma.base import _to_chroma_filter

from heavyrag.common.utils import delete_documents

logger = logging.getLogger(__name__)


def get_existing_tables_from_collection(collection: Collection) -> list[str]:
    """
    Gets the existing tables from ChromaDB collection.
    """
    dbname = collection.name
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

    tables: list[str] = list(set([i["table"] for i in query_result["metadatas"]]))  # type: ignore
    logger.debug(f"{len(tables)} tables already exists in the vectorstore collection")
    return tables


def delete_table_nodes(collection: Collection, tables: list[str]):
    """
    Helps to delete all the nodes associated with a table from the chroma collection.
    """
    dbname = collection.name
    filters = MetadataFilters(
        filters=[MetadataFilter(key="table", operator=FilterOperator.EQ, value=table) for table in tables],
        condition=FilterCondition.OR,
    )
    delete_documents(collection, filters)
    logger.debug(f'Deleted all nodes associated with the "{tables}" table from "{dbname}" collection...')
