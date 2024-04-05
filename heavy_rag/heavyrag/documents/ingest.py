from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.database import DatabaseReader

from heavyrag.documents.enums import DocMetadataKeys, DocType
from heavyrag.documents.index import get_or_create_document_index
from heavyrag.documents.pipeline import get_pipeline_by_document_type
from heavyrag.documents.utils import get_existing_doc_ids_from_collection


def sync_doc_index(reader: DatabaseReader, collection_name: str) -> VectorStoreIndex:
    """
    Syncs ChromaDB Vectorstore Document Index.
    """
    index = get_or_create_document_index(collection_name)
    chroma_collection = index._vector_store.client
    existing_ids_on_collection = set(get_existing_doc_ids_from_collection(chroma_collection))
    ids_joined = ",".join([f'"{i}"' for i in existing_ids_on_collection])
    documents = reader.load_data(
        f"select * from documents where id NOT IN ({ids_joined});"
    )  # where id not in existing_ids_on_collection
    nodes = []
    for doc in documents:
        doc_type = DocType._member_map_[doc.metadata[DocMetadataKeys.DOC_TYPE.value]]
        pipeline = get_pipeline_by_document_type(doc_type)  # type: ignore
        nodes.extend(pipeline.run(documents=[doc]))
    if nodes:
        index.insert_nodes(nodes=nodes)
    return index
