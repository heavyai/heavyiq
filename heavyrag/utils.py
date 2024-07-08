from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters, VectorStoreQuery

from heavyrag.filters import document_filters, facts_filters


async def get_nodes(
    index: VectorStoreIndex, filters: Optional[MetadataFilters] = None, limit: int | None = None, **kwargs
) -> list[NodeWithScore]:
    """
    Get the nodes by applying filters.
    """
    retriever = index.as_retriever(filters=filters, similarity_top_k=limit, **kwargs)  # how many nodes to return
    query = VectorStoreQuery(
        similarity_top_k=retriever._similarity_top_k,
        filters=retriever._filters,
    )
    query_result = await retriever._vector_store.aquery(query, **retriever._kwargs)
    nodes = retriever._build_node_list_from_query_result(query_result)
    return nodes


async def is_document_node_in_index(index: VectorStoreIndex) -> bool:
    """
    Checks for atleast a document node exists on the passed index.
    """
    if await get_nodes(index=index, filters=document_filters, limit=1):
        return True
    return False


async def is_facts_node_in_index(index: VectorStoreIndex) -> bool:
    """
    Checks for atleast a document node exists on the passed index.
    """
    if await get_nodes(index=index, filters=facts_filters, limit=1):
        return True
    return False
