from llama_index.core import VectorStoreIndex, load_index_from_storage
from heavyrag.read import HeavyDBReader
from llama_index.core import VectorStoreIndex
from heavyrag.index import get_or_create_index
from heavyrag.utils import get_existing_tables_from_collection


def sync_index(reader: HeavyDBReader) -> VectorStoreIndex:
    index = get_or_create_index()
    existing_tables = get_existing_tables_from_collection(
        index._vector_store.client, reader.database_name
    )
    # delete tables from the index if it not present on the database
    database_tables = reader.tables
    tables_to_delete_from_index = []
    for table in existing_tables:
        if table not in database_tables:
            tables_to_delete_from_index.append(table)

    # TODO: delete existing tables

    if sorted(reader.tables) != sorted(existing_tables):
        documents = reader.read_table_schemas(exclude_tables=existing_tables)
        for doc in documents:
            index.insert(doc)

    return index
