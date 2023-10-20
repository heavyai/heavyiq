import re
import pytest
from unittest.mock import patch
from langchain.agents import AgentExecutor
from heavyiq.langchain import HeavyDB
from langchain.schema import HumanMessage, AIMessage
from heavyiq.langchain.agents.convo_agent import create_conversational_agent
from tests import aoverride_config


@pytest.mark.asyncio
@aoverride_config
async def test_should_create_and_run_chat_sql_agent_which_saves_last_run_sql_into_memory(
    mock_heavy_db: HeavyDB, heavyiq_config
):
    agent: AgentExecutor = create_conversational_agent(mock_heavy_db)
    assert isinstance(agent, AgentExecutor)
    question = "What is the total population in the USA according to the data in the usa_states table?"
    response = await agent.arun(question)
    assert re.match(r"(?is).*?total population.*?\s+usa.*?\d+.*", response)
    last_run_sql_messages = agent._heavyiq_buffer_memory.load_memory_variables({})["last_run_sql"]  # type: ignore
    assert last_run_sql_messages
    # previous message to the last message should be HumanMessage
    human_message, ai_message = last_run_sql_messages[-2], last_run_sql_messages[-1]
    assert isinstance(human_message, HumanMessage)
    assert human_message.content == question
    assert isinstance(ai_message, AIMessage)
    # result column may or maynot get aliased
    assert re.search(r"(?i)SELECT SUM\(population\)(?: as \w+)? FROM usa_states;", ai_message.content)
