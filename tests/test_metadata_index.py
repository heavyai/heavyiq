import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch, AsyncMock

import pytest
from langchain_core.documents import Document

from heavyiq.langchain.index import HeavyDBMetadataIndex, aget_heavydb_index

# Mark entire module as integration test (requires working embedding server)
pytestmark = pytest.mark.integration


async def mock_aread_table_documents(include=None):
    """Mock table documents for testing."""
    docs = [
        Document(
            page_content="usa_states table contains information about US states including population, area, and capital cities.",
            metadata={"source": "usa_states"}
        ),
        Document(
            page_content="flights_2008 table contains flight data from 2008 including delays, cancellations, and routes.",
            metadata={"source": "flights_2008"}
        ),
    ]
    for doc in docs:
        if include is None or doc.metadata["source"] in include:
            yield doc


@pytest.fixture(scope="module")
async def metadata_index() -> AsyncIterator[HeavyDBMetadataIndex]:
    """
    Async metadata index fixture.
    """
    with patch(
        "heavyiq.langchain.index.heavydb.create_index.agenerate_table_documents",
        return_value=["usa_states", "flights_2008"],
    ), patch(
        "heavyiq.langchain.index.heavydb.create_index.aread_table_documents",
        side_effect=mock_aread_table_documents,
    ):
        result = await aget_heavydb_index()
        yield result


@pytest.mark.anyio
async def test_simple_search_for_table_names(heavyiq_config: Any, metadata_index: HeavyDBMetadataIndex):
    table_names = await metadata_index.asimple_search_for_table_names("How many states begin with the letter A?")
    assert "usa_states" in table_names
    table_names = await metadata_index.asimple_search_for_table_names(
        "What were the total number of flights that were cancelled in 2008?"
    )
    assert "flights_2008" in table_names


if __name__ == "__main__":
    unittest.main()
