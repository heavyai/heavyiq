from llama_index.core import VectorStoreIndex
from llama_index.readers.database import DatabaseReader

from heavyrag.documents.index import get_or_create_document_index
from heavyrag.documents.utils import get_existing_doc_ids_from_collection


def sync_doc_index(reader: DatabaseReader, collection_name: str) -> VectorStoreIndex:
    """
    Syncs ChromaDB Vectorstore Document Index.
    """
    index = get_or_create_document_index(collection_name)
    chroma_collection = index._vector_store.client
    existing_ids_on_collection = set(get_existing_doc_ids_from_collection(chroma_collection))
    documents = reader.load_data("select doc_id from documents.documents;")
    found_ids_on_database = set([i["doc_id"] for i in documents])
    ids_to_delete = existing_ids_on_collection - found_ids_on_database
    ids_to_include = found_ids_on_database - existing_ids_on_collection
    return index
