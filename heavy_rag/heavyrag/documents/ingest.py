from collections import defaultdict
from typing import Generator

from heavyrag.documents.enums import DocMetadataKeys, DocType
from heavyrag.documents.index import get_or_create_document_index
from heavyrag.documents.pipeline import get_pipeline_by_document_type
from heavyrag.documents.utils import get_existing_doc_ids_from_collection
from heavyrag.logging_conf import logger
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, Document
from llama_index.readers.database import DatabaseReader


def sync_doc_index(reader: DatabaseReader, collection_name: str) -> VectorStoreIndex:
    """
    Syncs ChromaDB Vectorstore Document Index.
    """
    logger.info("Started syncing document index.")
    index = get_or_create_document_index(collection_name)
    chroma_collection = index._vector_store.client
    existing_ids_on_collection = set(get_existing_doc_ids_from_collection(chroma_collection))
    ids_joined = ",".join([f'"{i}"' for i in existing_ids_on_collection])
    documents = reader.load_data(
        f"select * from documents where id NOT IN ({ids_joined});"
    )  # where id not in existing_ids_on_collection
    nodes: list[BaseNode] = []
    logger.info("Splitting documents into multiple nodes.")
    for pipeline, docs in get_docs_group_by_pipeline(documents):
        splitted_nodes = pipeline.run(documents=docs, show_progress=True)
        nodes.extend(splitted_nodes)
    logger.info("Inserting nodes into document index.")
    index.insert_nodes(nodes=nodes, show_progress=True)
    logger.info("Finished syncing document index!")
    return index


def get_docs_group_by_pipeline(
    documents: list[Document],
) -> Generator[tuple[IngestionPipeline, list[Document]], None, None]:
    """
    Yield docs group by pipeline.
    """
    type_pipeline_mapping: dict = {}
    type_docs_mapping: dict = defaultdict(list)
    for doc in documents:
        doc_type = DocType._member_map_[doc.metadata[DocMetadataKeys.DOC_TYPE.value]]
        if doc_type not in type_pipeline_mapping:
            pipeline = get_pipeline_by_document_type(doc_type)  # type: ignore
            type_pipeline_mapping[doc_type] = pipeline
        type_docs_mapping[doc_type].append(doc)

    for doc_type, docs in type_docs_mapping.items():
        pipe = type_pipeline_mapping[doc_type]
        yield pipe, docs


async def async_doc_index(reader: DatabaseReader, collection_name: str) -> VectorStoreIndex:
    """
    Syncs ChromaDB Vectorstore Document Index asynchronously.
    """
    logger.info("Started syncing document index async.")
    index = get_or_create_document_index(collection_name)
    chroma_collection = index._vector_store.client
    existing_ids_on_collection = set(get_existing_doc_ids_from_collection(chroma_collection))
    ids_joined = ",".join([f'"{i}"' for i in existing_ids_on_collection])
    documents = reader.load_data(
        f"select * from documents where id NOT IN ({ids_joined});"
    )  # where id not in existing_ids_on_collection
    nodes: list[BaseNode] = []
    logger.info("Splitting documents into multiple nodes.")

    for pipeline, docs in get_docs_group_by_pipeline(documents):
        splitted_nodes = await pipeline.arun(documents=docs, show_progress=True, num_workers=8)
        nodes.extend(splitted_nodes)

    logger.info("Inserting nodes into document index.")
    await index._async_add_nodes_to_index(
        index.index_struct,
        nodes,
        show_progress=True,
    )
    logger.info("Finished syncing document index!")
    return index
