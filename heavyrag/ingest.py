# Module for ingesting/inserting or deleting data into vectorDB
import asyncio
from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.base import BaseIndex

from heavyiq.langchain.heavydb import HeavyDB
from heavyrag.index import acreate_index_and_insert_nodes, get_index, get_or_create_index
from heavyrag.loaders import aload_file, aload_files_from_directory, aload_table, aload_tables
from heavyrag.query import any_table_node, delete_table_nodes, has_table_node
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


async def ainsert_tables(heavydb: HeavyDB) -> BaseIndex:
    """
    Insert data relevant to all tables in a database.
    """
    documents = await aload_tables(heavydb=heavydb)
    nodes = await atransform(documents=documents)
    index = get_or_create_index(collection_name=heavydb._dbname)
    index.insert_nodes(nodes=nodes, show_progress=True)
    return index


async def adelete_document(file_name: str, heavydb_name: str) -> None:
    """
    Delete all nodes assocuate with a particular document.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        return None
    index.vector_store._collection.delete(where={"name": file_name, "type": "document"})


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
