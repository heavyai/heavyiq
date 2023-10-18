import functools
from async_lru import alru_cache
from .generate_table_documents import (
    acreate_and_write_table_document,
)
from .create_index import (
    get_vectorstore_index_creator,
    aupdate_tables_in_index,  # don't remove these unused imports, there are several modules imported the relevant function from here
    acreate_index_if_nonexistent,
)
from .heavydb_metadata_index import HeavyDBMetadataIndex


@alru_cache
async def aget_heavydb_index() -> HeavyDBMetadataIndex:
    return await acreate_index_if_nonexistent()
