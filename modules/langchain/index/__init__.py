import functools

from .generate_table_documents import generate_table_documents
from .create_index import create_index_if_nonexistent
from .heavydb_metadata_index import HeavyDBMetadataIndex


@functools.cache
def get_heavydb_index() -> HeavyDBMetadataIndex:
    return create_index_if_nonexistent()
