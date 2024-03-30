from llama_index.core import VectorStoreIndex, load_index_from_storage

from heavyrag.index import get_or_create_index
from heavyrag.read import HeavyDBReader
from heavyrag.utils import delete_table_nodes, get_existing_tables_from_collection


def sync_index(reader: HeavyDBReader) -> VectorStoreIndex:
    """
    Syncs ChromaDB Vectorstore Index.
    """

    index = get_or_create_index(reader.database_name)
    chroma_collection = index._vector_store.client
    existing_tables = get_existing_tables_from_collection(chroma_collection)
    # delete tables from the index if it not present on the database
    database_tables = reader.tables
    tables_to_delete_from_index = []
    for table in existing_tables:
        if table not in database_tables:
            tables_to_delete_from_index.append(table)

    # delete existing tables
    if tables_to_delete_from_index:
        delete_table_nodes(chroma_collection, tables_to_delete_from_index)

    if sorted(reader.tables) != sorted(existing_tables):
        documents = reader.read_table_schemas(exclude_tables=existing_tables)
        for doc in documents:
            index.insert(doc)

    return index
