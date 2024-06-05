# Module for ingesting/inserting or deleting data into vectorDB
import asyncio
from collections import defaultdict

from chromadb import Collection
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from heavyiq.langchain.heavydb import HeavyDB
from heavyrag.index import acreate_index_and_insert_nodes, get_index, get_or_create_index
from heavyrag.loaders import aload_file, aload_files_from_directory, aload_table, aload_tables, load_fact, load_facts
from heavyrag.query import any_table_node, has_table_node
from heavyrag.transform import atransform


async def ainsert_documents(source_dir: str) -> list[BaseIndex]:
    """
    Grab documents from a source folder recursivley, split, transfor ingest into vectordb, finally create index.
    """
    documents = await aload_files_from_directory(source_dir)
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
        db_nodes = nodes_group_by_database_name[dbname]
        tasks.append(
            acreate_index_and_insert_nodes(db_nodes, collection_name=dbname, collection_metadata={"type": "document"})
        )

    return await asyncio.gather(*tasks)


async def ainsert_document(filepath: str, heavydb_name: str) -> BaseIndex:
    """
    Inserts a document into an existing or newly created index.
    """
    documents = await aload_file(filepath)
    nodes = await atransform(documents=documents)
    index = get_index(collection_name=heavydb_name)
    if not index:
        # index not found so create a new one
        index = await acreate_index_and_insert_nodes(nodes=nodes, collection_name=heavydb_name)
    else:
        # index already exists, so just add nodes to it
        index.insert_nodes(nodes=nodes, show_progress=True)

    return index


async def ainsert_table(heavydb: HeavyDB, table_name: str) -> BaseIndex:
    """
    Inserts information related to a HeavyDB table.
    """
    documents = await aload_table(heavydb=heavydb, table_name=table_name)
    nodes = await atransform(documents=documents)
    index = get_or_create_index(collection_name=heavydb._dbname)
    index.insert_nodes(nodes=nodes, show_progress=True)
    return index


async def get_or_create_index_and_insert_nodes(collection_name: str, nodes: list[BaseNode]) -> BaseIndex:
    """
    Added nodes to the existing or newly created index.
    """
    index = get_or_create_index(collection_name=collection_name)
    index.insert_nodes(nodes=nodes, show_progress=True)
    return index


async def ainsert_tables(heavydb: HeavyDB) -> BaseIndex:
    """
    Insert data relevant to all tables in a database.
    """
    documents = await aload_tables(heavydb=heavydb)
    nodes = await atransform(documents=documents)
    index = get_or_create_index(collection_name=heavydb._dbname)
    index.insert_nodes(nodes=nodes, show_progress=True)
    return index


async def ainsert_fact(fact_id: str, fact: str, heavydb_name: str) -> BaseIndex:
    """
    Form a textnode and then insert it into the index.
    """
    fact_node = load_fact(fact_id=fact_id, fact=fact, heavydb_name=heavydb_name)
    index = get_or_create_index(collection_name=heavydb_name)
    index.insert_nodes(nodes=[fact_node], show_progress=True)
    return index


async def aupdate_fact(fact_id: str, fact: str, heavydb_name: str) -> BaseIndex:
    """
    Update fact.
    """
    index = get_or_create_index(collection_name=heavydb_name)
    # deleet the relevant node by id
    delete_node_by_id(index=index, id=fact_id)
    # re-insert the node once again
    fact_node = load_fact(fact_id=fact_id, fact=fact, heavydb_name=heavydb_name)
    index.insert_nodes(nodes=[fact_node], show_progress=True)
    return index


async def adelete_fact(fact_id: str, heavydb_name: str) -> BaseIndex:
    """
    Delete a particular fact by fact_id.
    """
    index = get_or_create_index(collection_name=heavydb_name)
    # deleet the relevant node by id
    delete_node_by_id(index=index, id=fact_id)
    return index


async def adelete_facts(heavydb_name: str) -> BaseIndex:
    """
    Delete a particular fact by fact_id.
    """
    index = get_or_create_index(collection_name=heavydb_name)
    # deleet the relevant node by id
    await adelete_database_facts(index=index, heavydb_name=heavydb_name)
    return index


async def ainsert_facts(facts: str, heavydb: HeavyDB) -> BaseIndex:
    """
    Insert facts nodes.
    """
    documents = await load_facts(facts=facts, heavydb_name=heavydb._dbname)
    pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=120, chunk_overlap=10)])
    nodes = await atransform(documents=documents, pipeline=pipeline)
    return await get_or_create_index_and_insert_nodes(collection_name=heavydb._dbname, nodes=nodes)


async def upsert_facts(facts: str, heavydb: HeavyDB) -> None:
    """
    This supposed to delete and insert facts.
    """
    index = get_or_create_index(collection_name=heavydb._dbname)
    await adelete_database_facts(index, heavydb_name=heavydb._dbname)
    await ainsert_facts(facts=facts, heavydb=heavydb)


def delete_nodes(collection: Collection, where: dict, batch_size: int = 166) -> bool:
    """
    Helps to delete nodes.
    """
    ids = collection.get(where=where)["ids"]
    if not ids:
        return False

    doc_count = len(ids)
    if doc_count < batch_size:
        collection.delete(ids=ids)
        return True

    for i in range(0, len(ids), batch_size):
        batched_ids = ids[i : i + batch_size]
        collection.delete(ids=batched_ids)

    return True


async def adelete_document(file_name: str, heavydb_name: str) -> bool:
    """
    Delete all nodes associate with a particular document.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        return False

    collection = index.vector_store._collection
    return delete_nodes(collection=collection, where={"$and": [{"name": file_name}, {"type": "document"}]})


async def delete_table_nodes(index: VectorStoreIndex, table_name: str) -> bool:
    """
    Delete all the nodes associated with a table.
    """
    collection = index.vector_store._collection
    return delete_nodes(collection=collection, where={"$and": [{"name": table_name}, {"type": "table"}]})


def delete_nodes_by_ids(index: VectorStoreIndex, ids: list[str]) -> bool:
    """
    Helps to delete nodes by "id".
    """
    collection = index.vector_store._collection
    return delete_nodes(collection=collection, where={"id": {"$in": ids}})


def delete_node_by_id(index: VectorStoreIndex, id: str) -> bool:
    """
    Helps to delete nodes by "id".
    """
    collection = index.vector_store._collection
    return delete_nodes(collection=collection, where={"id": {"$eq": id}})


async def adelete_database_facts(index: VectorStoreIndex, heavydb_name: str) -> bool:
    """
    Delete all facts nodes associated with a heavydb database.
    """
    collection = index.vector_store._collection
    return delete_nodes(collection=collection, where={"$and": [{"dbname": heavydb_name}, {"type": "facts"}]})


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
