import re
from unittest.mock import patch

import nest_asyncio
import pytest

from heavyiq.config import HeavyIQConfig
from heavyiq.langchain import HeavyDB

# always apply nest_asyncio when the module contain more than one async test cases
nest_asyncio.apply()


@pytest.mark.asyncio
async def test_aget_single_table_info_should_pass_for_inline_column_metadata(
    mock_heavy_db: HeavyDB,
    heavyiq_config: HeavyIQConfig,
):
    tablename = "heavyai_us_states"
    heavyiq_config.inline_column_metadata_on_table_info_prompt = True
    with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
        out = await mock_heavy_db.aget_single_table_info(tablename)
        # out = event_loop.run_until_complete(mock_heavy_db.aget_single_table_info(tablename))
        # this includes sample rows, top-k, time-range and comments
        # ensure sample rows
        assert f"rows from {tablename} table:" in out
        # ensure top-k values are shown inline (e.g., "id TEXT (05, 06, 02 ...)")
        assert re.search(r"(?m)^id TEXT \(\d+(?:, \d+)* \.{3}\)", out)


@pytest.mark.asyncio
async def test_aget_single_table_info_should_pass_for_not_inline_column_metadata(
    mock_heavy_db: HeavyDB,
    heavyiq_config: HeavyIQConfig,
):
    tablename = "heavyai_us_states"
    heavyiq_config.inline_column_metadata_on_table_info_prompt = False
    with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
        # out = event_loop.run_until_complete(mock_heavy_db.aget_single_table_info(tablename, include_samples=False))
        out = await mock_heavy_db.aget_single_table_info(tablename, include_samples=False)
        # this includes top-k, time-range and comments
        # ensure top-k in seprate line
        assert "cardinality columns" in out
