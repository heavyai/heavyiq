import re
from unittest.mock import patch

from langchain.agents import AgentExecutor

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.agents.chat_agent import create_sql_agent
from tests import override_config


@override_config
def test_should_create_and_run_chat_sql_agent(heavyiq_config, mock_heavy_db: HeavyDB):
    agent = create_sql_agent(mock_heavy_db)
    assert isinstance(agent, AgentExecutor)
    response = agent.run("What is the total population in the USA according to the data in the usa_states table?")
    assert re.match(r"(?s).*?total population.*?\s+usa_states\s+.*?\d+.*", response)
