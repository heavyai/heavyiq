# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# test cases for heavyrag.main
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from llama_index.core.schema import TextNode

from heavyiq.langchain.heavydb import HeavyDB

pytestmark = pytest.mark.heavydb


@pytest.mark.anyio
async def test_determine_relevant_table_names():
    """
    Test RAG retrieval for relevant table names based on the given question.
    """
    from heavyrag.main import determine_table_names

    overture_db_name, cars_db_name = "overture_wa", "cars"
    overture_db = await HeavyDB.from_env_async(db_name=overture_db_name)
    cars_db = await HeavyDB.from_env_async(db_name=cars_db_name)

    names = await determine_table_names(question="How many addresses are there?", heavydb=overture_db, force_sync=True)
    names = await determine_table_names(
        question="Which country produces more number of cars?", heavydb=cars_db, force_sync=True
    )
    assert "production" in names and "overture_buildings_wa" not in names
