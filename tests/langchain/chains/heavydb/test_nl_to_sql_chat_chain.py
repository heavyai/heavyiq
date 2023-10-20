import pytest
from heavyiq.langchain import HeavyDB
from tests import aoverride_config
from heavyiq.langchain.logging import log_chain_call_async
from heavyiq.langchain.chains import NLtoSQLChatChain
from tests.langchain import FakeChatOpenAI


@pytest.mark.asyncio
@aoverride_config
async def test_nl_to_sql_chat_async_chain_should_pass(
    mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI, heavyiq_config
):
    chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db)
    chain_input = {chain.input_key: "", "tables": ["usa_states"]}
    res = await log_chain_call_async(chain, chain_input, "")
    chain_output = res[chain.output_key]
    assert chain_output == "SELECT SUM(POPULATION) AS total_population\nFROM usa_states;"
