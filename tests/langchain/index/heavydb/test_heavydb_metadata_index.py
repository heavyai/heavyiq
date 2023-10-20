import pytest
from tests import aoverride_config
from langchain.schema import Document
from heavyiq.langchain.index.heavydb.heavydb_metadata_index import HeavyDBMetadataIndex


@pytest.mark.asyncio
@aoverride_config
async def test_heavydb_metadata_index_should_do_a_simple_search_for_tables_docs(
    heavydb_metadata_index: HeavyDBMetadataIndex, heavyiq_config
):
    assert await heavydb_metadata_index.simple_search_for_table_docs("movie_actress")


@pytest.mark.asyncio
@aoverride_config
async def test_heavydb_simple_search_for_table_names_docs(heavydb_metadata_index: HeavyDBMetadataIndex, heavyiq_config):
    assert "movie_actress" in await heavydb_metadata_index.simple_search_for_table_names("movie_actress")
