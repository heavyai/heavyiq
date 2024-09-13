# Module for ingesting/inserting or deleting data into vectorDB
import asyncio
from collections import defaultdict

from chromadb import Collection
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma.base import _to_chroma_filter

from heavyiq.langchain.heavydb import HeavyDB
from heavyrag.chroma_ops import update_embeddings
from heavyrag.index import acreate_index_and_insert_nodes, get_index, get_or_create_index
from heavyrag.loaders import aload_file, aload_files_from_directory, aload_table, aload_tables, load_fact, load_facts
from heavyrag.logger import logger
from heavyrag.query import any_table_node, has_table_node
from heavyrag.transform import atransform


async def ainsert_documents(source_dir: str) -> list[BaseIndex]:
    """
    Grab documents from a source folder recursivley, split, transfor ingest into vectordb, finally create index.
    """
    logger.debug(f"Started loading files for document insertion from the {source_dir} directory")
    documents = await aload_files_from_directory(source_dir)
    logger.debug("Loaded documents, transforming each document into nodes...")
    nodes = await atransform(documents=documents)
    # group the nodes by database id
    nodes_group_by_database_name = defaultdict(list)
    for node in nodes:
        try:
            nodes_group_by_database_name[node.metadata["dbname"]].append(node)
        except KeyError:
            print(node.get_content())
            pass

    tasks = []
    for dbname in nodes_group_by_database_name:
        logger.debug(f"Inserting nodes into {dbname} index.")
        db_nodes = nodes_group_by_database_name[dbname]
        tasks.append(
            acreate_index_and_insert_nodes(db_nodes, collection_name=dbname, collection_metadata={"type": "document"})
        )

    return await asyncio.gather(*tasks)


async def ainsert_document(filepath: str, heavydb_name: str) -> BaseIndex:
    """
    Inserts a document into an existing or newly created index.
    """
    logger.debug(f"Index: Inserting a single document {filepath}...")
    documents = await aload_file(filepath)
    nodes = await atransform(documents=documents)
    index = get_index(collection_name=heavydb_name)
    if not index:
        # index not found so create a new one
        index = await acreate_index_and_insert_nodes(nodes=nodes, collection_name=heavydb_name)
    else:
        # index already exists, so just add nodes to it
        index.insert_nodes(nodes=nodes, show_progress=True)

    logger.debug(f"Successfully inserted all the nodes relevant to {filepath} document.")

    return index


async def ainsert_table(heavydb: HeavyDB, table_name: str) -> BaseIndex:
    """
    Inserts information related to a HeavyDB table.
    """
    logger.debug(f"Index: Inserting {table_name} table information into {heavydb} index/collection.")
    documents = await aload_table(heavydb=heavydb, table_name=table_name)
    nodes = await atransform(documents=documents)
    index = get_or_create_index(collection_name=heavydb._dbname)
    index.insert_nodes(nodes=nodes, show_progress=True)
    logger.debug(f"Successfully inserted {table_name} table information.")
    return index


async def update_table(heavydb: HeavyDB, table_name: str) -> BaseIndex:
    """
    Update table information on RAG index.
    """
    logger.debug(f"Index: Updating {table_name} table information on {heavydb} index/collection.")
    index = get_or_create_index(collection_name=heavydb._dbname)
    # delete existing table information
    await delete_table_nodes(index, table_name)
    # re-insert the latest schema
    documents = await aload_table(heavydb=heavydb, table_name=table_name)
    nodes = await atransform(documents=documents)
    index.insert_nodes(nodes=nodes, show_progress=True)
    logger.debug(f"Successfully updated {table_name} table information.")
    return index


async def get_or_create_index_and_insert_nodes(collection_name: str, nodes: list[BaseNode]) -> BaseIndex:
    """
    Added nodes to the existing or newly created index.
    """
    logger.debug(f"Index: Inserting {len(nodes)} nodes into {collection_name} collection.")
    index = get_or_create_index(collection_name=collection_name)
    index.insert_nodes(nodes=nodes, show_progress=True)
    logger.debug(f"Successfully inserted {len(nodes)} nodes.")
    return index


async def ainsert_tables(heavydb: HeavyDB) -> BaseIndex:
    """
    Insert data relevant to all tables in a database.
    """
    logger.info(f"Index: Bulk Inserting table info into {heavydb} collection.")
    documents = await aload_tables(heavydb=heavydb)
    nodes = await atransform(documents=documents)
    index = get_or_create_index(collection_name=heavydb._dbname)
    index.insert_nodes(nodes=nodes, show_progress=True)
    logger.info(f"Successfully Bulk Inserted table info into {heavydb._dbname} collection.")
    return index


async def ainsert_fact(fact_id: str, fact: str, heavydb_name: str) -> BaseIndex:
    """
    Form a textnode and then insert it into the index.
    """
    fact_node = load_fact(fact_id=fact_id, fact=fact, heavydb_name=heavydb_name)
    index = get_or_create_index(collection_name=heavydb_name)
    index.insert_nodes(nodes=[fact_node], show_progress=True)
    logger.info(f"Index: Inserted a fact into {heavydb_name} collection.")
    return index


async def aupdate_fact(fact_id: str, fact: str, heavydb_name: str) -> BaseIndex:
    """
    Update fact.
    """
    index = get_index(collection_name=heavydb_name)
    # deleet the relevant node by id
    delete_node_by_id(index=index, id=fact_id)
    # re-insert the node once again
    fact_node = load_fact(fact_id=fact_id, fact=fact, heavydb_name=heavydb_name)
    index.insert_nodes(nodes=[fact_node], show_progress=True)
    logger.info(f"Index: Updated a fact ({fact_id}) on {heavydb_name} collection.")
    return index


async def adelete_fact(fact_id: str, heavydb_name: str) -> BaseIndex:
    """
    Delete a particular fact by fact_id.
    """
    index = get_index(collection_name=heavydb_name)
    # deleet the relevant node by id
    delete_node_by_id(index=index, id=fact_id)
    logger.info(f"Index: Deleted a fact ({fact_id}) from {heavydb_name} collection.")
    return index


async def adelete_facts(heavydb_name: str, fact_ids: list[str] | None = None) -> None | BaseIndex:
    """
    Delete a all database facts or specific facts.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        return None
    if fact_ids:
        # delete only the facts associated with the passed facts ids
        await adelete_facts_by_ids(index=index, heavydb_name=heavydb_name, facts_ids=fact_ids)
    else:
        # delete all the facts relevant to a particular database
        await adelete_database_facts(index=index, heavydb_name=heavydb_name)
        logger.info(f"Index: Deleted all facts from {heavydb_name} collection.")
    return index


async def ainsert_facts(facts: list[tuple[str, str]], heavydb_name: str) -> BaseIndex:
    """
    Insert facts nodes.
    """
    logger.info(f"Index: Bulk inserting facts into {heavydb_name} collection.")
    nodes = load_facts(facts=facts, heavydb_name=heavydb_name)
    return await get_or_create_index_and_insert_nodes(collection_name=heavydb_name, nodes=nodes)


def delete_nodes(collection: Collection, where: dict, batch_size: int = 166) -> bool:
    """
    Helps to delete nodes.
    """
    ids = collection.get(where=where)["ids"]
    if not ids:
        return False

    logger.debug(f"Index: Deleting the following node ids: {ids}")

    doc_count = len(ids)
    if doc_count < batch_size:
        collection.delete(ids=ids)
        return True

    for i in range(0, len(ids), batch_size):
        batched_ids = ids[i : i + batch_size]
        collection.delete(ids=batched_ids)

    return True


def re_embed_documents(collection_name: str) -> bool:
    """
    Re-embed all the documents available in a chromadb collection.
    This involves collection backup, collection deletion and collection restore with new embeddings.
    """
    idx = get_index(collection_name)
    if not idx:
        return False

    collection = idx.vector_store._collection  # type: ignore
    update_embeddings(collection=collection)
    return True


async def adelete_document(file_name: str, heavydb_name: str) -> bool:
    """
    Delete all nodes associate with a particular document.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        return False

    collection = index.vector_store._collection  # type: ignore
    return delete_nodes(collection=collection, where={"$and": [{"name": file_name}, {"type": "document"}]})


async def delete_table_nodes(index: VectorStoreIndex, table_name: str) -> bool:
    """
    Delete all the nodes associated with a table.
    """
    collection = index.vector_store._collection  # type: ignore
    return delete_nodes(collection=collection, where={"$and": [{"name": table_name}, {"type": "table"}]})


def delete_nodes_by_ids(index: VectorStoreIndex, ids: list[str]) -> bool:
    """
    Helps to delete nodes by "id".
    """
    collection = index.vector_store._collection  # type: ignore
    return delete_nodes(collection=collection, where={"id": {"$in": ids}})


def delete_node_by_id(index: VectorStoreIndex, id: str) -> bool:
    """
    Helps to delete nodes by "id".
    """
    collection = index.vector_store._collection  # type: ignore
    return delete_nodes(collection=collection, where={"id": {"$eq": id}})


async def adelete_database_facts(index: VectorStoreIndex, heavydb_name: str) -> bool:
    """
    Delete all facts nodes associated with a heavydb database.
    """
    collection = index.vector_store._collection  # type: ignore
    return delete_nodes(collection=collection, where={"$and": [{"dbname": heavydb_name}, {"type": "facts"}]})


async def adelete_facts_by_ids(index: VectorStoreIndex, heavydb_name: str, facts_ids: list[str]) -> None:
    """
    Delete specific facts.
    """
    collection = index.vector_store._collection  # type: ignore
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="type", operator=FilterOperator.EQ, value="facts"),
            MetadataFilter(key="dbname", operator=FilterOperator.EQ, value=heavydb_name),
            MetadataFilter(key="id", operator=FilterOperator.IN, value=facts_ids),
        ],
        condition=FilterCondition.AND,
    )
    return delete_nodes(collection=collection, where=_to_chroma_filter(filters))


async def sync_table_index(heavydb: HeavyDB, force_sync: bool = False) -> VectorStoreIndex:
    """
    Supposed to sync a table index.
    """
    # check for collection exists
    index = get_index(collection_name=heavydb._dbname)
    if not index:
        # collection doesn't exists, so create one and do the nodes ingestion
        index = await ainsert_tables(heavydb=heavydb)
    else:
        # collection exists
        # so check for any table node exists
        any_table_node_exists = await any_table_node(index)
        if any_table_node_exists:
            # table node exists
            tables = heavydb.get_usable_table_names()
            # do force sync if it asks for
            if force_sync:
                # iterate over all the tables
                # delete all the nodes specifc to a particulatr table
                # re-fetch table info and then re-ingest it
                for table in tables:
                    await delete_table_nodes(index=index, table_name=table)
                # re-fetch and re-insert all
                index = await ainsert_tables(heavydb=heavydb)
            else:
                # iterate over all the tables
                # fetch table info and then ingest it only if there wan't any single table node associated
                # else skip it
                exclude_tables = []
                for table in tables:
                    if await has_table_node(index=index, table_name=table):
                        exclude_tables.append(table)

                documents = await aload_tables(heavydb=heavydb, exlude_tables=exclude_tables)
                nodes = await atransform(documents=documents)
                index.insert_nodes(nodes=nodes, show_progress=True)
        else:
            # no table node exists, so create table nodes for all the database tables
            index = await ainsert_tables(heavydb=heavydb)

    return index
