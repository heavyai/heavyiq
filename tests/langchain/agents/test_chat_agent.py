import re
import pytest
from unittest.mock import patch

from langchain.agents import AgentExecutor

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.agents.chat_agent import create_sql_agent
from tests import aoverride_config


@pytest.mark.asyncio
@aoverride_config
async def test_should_create_and_run_chat_sql_agent(mock_heavy_db: HeavyDB, heavyiq_config):
    agent = create_sql_agent(mock_heavy_db)
    assert isinstance(agent, AgentExecutor)
    response = await agent.arun(
        "What is the total population in the USA according to the data in the usa_states table?"
    )
    assert re.match(r"(?s).*?total population.*?\s+usa_states\s+.*?\d+.*", response)
