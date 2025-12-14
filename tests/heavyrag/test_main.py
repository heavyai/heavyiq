# test cases for heavyrag.main
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from heavyiq.langchain.heavydb import HeavyDB

if TYPE_CHECKING:
    from heavyrag.controller import BaseController


@pytest.mark.integration
@pytest.mark.anyio
async def test_determine_relevant_table_names(faiss_controller_patch: "BaseController"):
    """
    Test RAG retrieval for relevant table names based on the given question.
    Requires: HeavyDB, embedding server
    
    Available tables in heavyai db:
    - heavyai_us_states
    - heavyai_us_counties  
    - heavyai_countries
    """
    from heavyrag.main import determine_table_names

    db_name = "heavyai"
    heavydb = await HeavyDB.from_env_async(db_name=db_name)

    # Patch rag_controller to use the FAISS controller from fixture
    with patch("heavyrag.controller.rag_controller", faiss_controller_patch):
        # Question about US states should retrieve heavyai_us_states
        state_names = await determine_table_names(
            question="How many US states are there?",
            heavydb=heavydb,
            force_sync=True
        )
        assert isinstance(state_names, list)
        assert "heavyai_us_states" in state_names

        # Question about counties should retrieve heavyai_us_counties
        county_names = await determine_table_names(
            question="Show me all the counties in the United States",
            heavydb=heavydb,
            force_sync=False  # Already synced above
        )
        assert isinstance(county_names, list)
        assert "heavyai_us_counties" in county_names

        # Question about countries should retrieve heavyai_countries
        country_names = await determine_table_names(
            question="List all countries in the world",
            heavydb=heavydb,
            force_sync=False
        )
        assert isinstance(country_names, list)
        assert "heavyai_countries" in country_names
