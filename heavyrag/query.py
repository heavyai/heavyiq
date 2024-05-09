# Module helps to query the underlying vectorstore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQuery

from heavyrag.filters import get_document_filter_matches, get_table_filter_matches, table_filter


async def has_table_node(index: VectorStoreIndex, table_name: str) -> bool:
    """
    Check if there's any table_node exists for the passed table_name.
    """
    query = VectorStoreQuery(similarity_top_k=1, filters=get_table_filter_matches(table_name=table_name))
    result = await index.vector_store.aquery(query=query)
    if result.ids:
        return True
    return False


async def get_table_node_ids(index: VectorStoreIndex, table_name: str) -> list[str]:
    """
    Check if there's any table_node exists for the passed table_name.
    """
    query = VectorStoreQuery(similarity_top_k=1, filters=get_table_filter_matches(table_name=table_name))
    result = await index.vector_store.aquery(query=query)
    return result.ids


async def delete_table_nodes(index: VectorStoreIndex, table_name: str) -> None:
    """
    Delete all the nodes associated with a table.
    """
    index.vector_store._collection.delete(where={"name": table_name, "type": "table"})


async def has_document_node(index: VectorStoreIndex, file_name: str) -> bool:
    """
    Check if there's any docuemnt node exists for the passed file_name.
    """
    query = VectorStoreQuery(similarity_top_k=1, filters=get_document_filter_matches(file_name=file_name))
    result = await index.vector_store.aquery(query=query)
    if result.ids:
        return True
    return False


async def any_table_node(index: VectorStoreIndex) -> bool:
    """
    Check if there's any table node exists on the whole vectorstore.
    """
    query = VectorStoreQuery(similarity_top_k=1, filters=MetadataFilters(filters=[table_filter]))
    result = await index.vector_store.aquery(query=query)
    if result.ids:
        return True
    return False
