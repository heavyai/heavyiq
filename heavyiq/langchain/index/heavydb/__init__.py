import functools
from async_lru import alru_cache
from .generate_table_documents import (
    generate_table_documents,
    create_and_write_table_document,
    acreate_and_write_table_document,
)
from .create_index import (
    create_index_if_nonexistent,
    get_vectorstore_index_creator,
    update_tables_in_index,
    acreate_index_if_nonexistent,
)
from .heavydb_metadata_index import HeavyDBMetadataIndex


@functools.cache
def get_heavydb_index() -> HeavyDBMetadataIndex:
    return create_index_if_nonexistent()


@alru_cache
async def aget_heavydb_index() -> HeavyDBMetadataIndex:
    return await acreate_index_if_nonexistent()
