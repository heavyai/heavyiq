import functools

from async_lru import alru_cache

from .create_index import (  # don't remove these unused imports, there are several modules imported the relevant function from here
    acreate_index_if_nonexistent,
    aupdate_tables_in_index,
    create_index_if_nonexistent,
    get_vectorstore_index_creator,
    update_tables_in_index,
)
from .generate_table_documents import (
    acreate_and_write_table_document,
    create_and_write_table_document,
    generate_table_documents,
)
from .heavydb_metadata_index import HeavyDBMetadataIndex


@functools.cache
def get_heavydb_index() -> HeavyDBMetadataIndex:
    return create_index_if_nonexistent()


@alru_cache()
async def aget_heavydb_index(session: str | None = None) -> HeavyDBMetadataIndex:
    return await acreate_index_if_nonexistent(session=session)
