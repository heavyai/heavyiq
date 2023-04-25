import functools

from .generate_table_documents import generate_table_documents, create_and_write_table_document
from .create_index import create_index_if_nonexistent, get_vectorstore_index_creator, update_tables_in_index
from .heavydb_metadata_index import HeavyDBMetadataIndex


@functools.cache
def get_heavydb_index() -> HeavyDBMetadataIndex:
    return create_index_if_nonexistent()
