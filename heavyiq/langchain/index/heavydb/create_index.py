from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from chromadb.api import Where
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.base import VectorStore

from heavyiq.utils import is_path_exists
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.config import get_config
from ..utils import get_vectorstore_index_creator
from .generate_table_documents import agenerate_table_documents
from .heavydb_metadata_index import HeavyDBMetadataIndex
from .utils import aread_table_documents


async def aupdate_tables_in_index(
    index_creator: VectorstoreIndexCreator, vectorstore: Chroma, tables_to_update: list[str]
):
    config = get_config()
    logger = get_heavyiq_logger()
    huggingface_model_name = config.huggingface_embed_model
    logger.info("Started updating tables in index.")
    logger.debug("New table summaries will replace existing summaries in the index (if any).")
    logger.info("Reading new table documents...")
    docs = [doc async for doc in aread_table_documents(include=tables_to_update)]
    if len(docs) != len(tables_to_update):
        raise ValueError("Not all provided tables have documents. Aborting.")
    logger.info("Deleting existing table documents from index...")
    where: Where | None = None
    if len(tables_to_update) == 1:
        where = {"source": tables_to_update[0]}  # type: ignore
        await run_in_threadpool(vectorstore._collection.delete, where=where)
    else:
        where = {"$or": [{"source": table_name} for table_name in tables_to_update]}
        await run_in_threadpool(vectorstore._collection.delete, where=where)
    logger.debug("Splitting documents...")
    sub_docs = index_creator.text_splitter.split_documents(docs)
    logger.debug(f"Indexing documents with {huggingface_model_name}...")
    await run_in_threadpool(vectorstore.add_documents, sub_docs)
    logger.debug("Done")


async def acreate_index_if_nonexistent() -> HeavyDBMetadataIndex:
    """
    Create a new vector store index if it does not exist, otherwise return the existing index asynchronously.

    :return: A HeavyDBMetadataIndex instance containing the vector store index.
    """
    logger = get_heavyiq_logger()
    config = get_config()
    huggingface_model_name = config.huggingface_embed_model
    metadata_index_dir = config.metadata_index_dir
    tables_with_new_summaries = await agenerate_table_documents()

    index_creator = get_vectorstore_index_creator(metadata_index_dir)

    vectorstore: VectorStore
    if await is_path_exists(metadata_index_dir):
        logger.debug("HeavyDB Index already exists. Returning existing index.")
        vectorstore = Chroma(embedding_function=index_creator.embedding, persist_directory=metadata_index_dir)
        if len(tables_with_new_summaries) > 0:
            await aupdate_tables_in_index(index_creator, vectorstore, tables_with_new_summaries)
    else:
        logger.debug("HeavyDB Index does not exist. Creating new index.")
        logger.debug("Reading table documents...")
        docs = [doc async for doc in aread_table_documents()]

        logger.debug("Splitting documents...")
        sub_docs = index_creator.text_splitter.split_documents(docs)
        logger.debug(f"Indexing documents with {huggingface_model_name}...")
        # there is no async version of chroma.from_documents method. so wrap it in run_in_threadpool.
        vectorstore = await run_in_threadpool(
            index_creator.vectorstore_cls.from_documents,
            sub_docs,
            index_creator.embedding,
            **index_creator.vectorstore_kwargs,
        )
        logger.debug("Done")

    return HeavyDBMetadataIndex(vectorstore=vectorstore, text_splitter=index_creator.text_splitter)  # type: ignore
