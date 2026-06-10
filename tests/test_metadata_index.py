# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from heavyiq.langchain.index import HeavyDBMetadataIndex, aget_heavydb_index

pytestmark = pytest.mark.heavydb


@pytest.fixture(scope="module")
async def metadata_index() -> AsyncIterator[HeavyDBMetadataIndex]:
    """
    Async metadata index fixture.
    """
    # this should generate missing documents (only when table_documents been emptied) and then
    # populates the index with those documents
    # if the index already exists then the documents related to "usa_states", "flights_2008" tables
    # gets updated
    with patch(
        "heavyiq.langchain.index.heavydb.create_index.agenerate_table_documents",
        return_value=["usa_states", "flights_2008"],
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
