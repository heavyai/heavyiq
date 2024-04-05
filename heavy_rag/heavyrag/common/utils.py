from chromadb.api.models.Collection import Collection
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.vector_stores.chroma.base import _to_chroma_filter


def chunk_list(lst, chunk_size: int = 166):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def delete_documents(collection: Collection, filters: MetadataFilters):
    """
    Helps to delete documents in chunks.
    """
    where = _to_chroma_filter(filters)
    result = collection.get(where=where)
    for chunk in chunk_list(result["ids"]):
        collection.delete(ids=chunk)
