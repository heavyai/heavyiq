import unittest
from typing import AsyncIterator

import pytest

from heavyiq.langchain.index import HeavyDBMetadataIndex, aget_heavydb_index


@pytest.fixture(scope="module")
async def metadata_index() -> AsyncIterator[HeavyDBMetadataIndex]:
    """
    Async metadata index fixture.
    """
    # this should generate missing documents and then
    # populates the index with those documents
    result = await aget_heavydb_index()
    yield result


@pytest.mark.anyio
async def test_simple_search_for_table_names(heavyiq_config, metadata_index: HeavyDBMetadataIndex):
    table_names = await metadata_index.asimple_search_for_table_names(
        "How many states begin with the letter A?", allowable_tables=["usa_states"]
    )
    assert table_names
    assert "usa_states" in table_names


if __name__ == "__main__":
    unittest.main()
