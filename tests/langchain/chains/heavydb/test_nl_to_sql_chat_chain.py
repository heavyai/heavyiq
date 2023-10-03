import pytest
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.logging import log_chain_call_async, log_chain_call
from heavyiq.langchain.chains import NLtoSQLChatChain
from tests.langchain import FakeChatOpenAI


@pytest.mark.asyncio
async def test_nl_to_sql_chat_async_chain_should_pass(mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI):
    chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db)
    chain_input = {chain.input_key: "", "tables": ["usa_states"]}
    res = await log_chain_call_async(chain, chain_input, "")
    chain_output = res[chain.output_key]
    assert chain_output == "SELECT SUM(POPULATION) AS total_population\nFROM usa_states;"


def test_nl_to_sql_chat_sync_chain_should_pass(mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI):
    chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db)
    chain_input = {chain.input_key: "", "tables": ["usa_states"]}
    res = log_chain_call(chain, chain_input, "")
    chain_output = res[chain.output_key]
    assert chain_output == "SELECT SUM(POPULATION) AS total_population\nFROM usa_states;"
