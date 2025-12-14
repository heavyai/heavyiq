import uuid
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
    
    # Build NodeWithScore list from query result
    nodes: list[NodeWithScore] = []
    if query_result.nodes:
        similarities = query_result.similarities or [1.0] * len(query_result.nodes)
        for node, score in zip(query_result.nodes, similarities):
            nodes.append(NodeWithScore(node=node, score=score))
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


async def get_document_node_count_in_index(index: VectorStoreIndex) -> int:
    """
    Get the total number of node count of type document.
    """
    nodes = await get_nodes(index=index, filters=document_filters, limit=None)
    return len(nodes)


async def get_facts_node_count_in_index(index: VectorStoreIndex) -> int:
    """
    Get the total number of node count of type document.
    """
    nodes = await get_nodes(index=index, filters=facts_filters, limit=None)
    return len(nodes)


def convert_uuid_str_to_hex(uuid_str: str) -> str:
    """
    Converts uuid str to hex.
    """
    uid = uuid.UUID(uuid_str)
    return uid.hex


# Convert UUID4 hex string to a 64-bit integer
def uuid4_hex_to_int64(uuid_hex: str) -> int:
    # Use the first 16 characters of the hex string (64 bits)
    return int(uuid_hex[:10], 16)


# Convert a 64-bit integer back to a hex string (partial UUID)
def int64_to_uuid4_hex(int64_val: int) -> str:
    # Convert the int64 back to a hex string (16 characters)
    return format(int64_val, "010x")
