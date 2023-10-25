import pytest
from heavyiq.langchain import HeavyDB
from tests import aoverride_config
from heavyiq.langchain.logging import log_chain_call_async
from heavyiq.langchain.chains import NLtoAnswerChain, NLtoSQLChatChain
from tests.langchain import FakeChatOpenAI, FakeOpenAI


@pytest.mark.asyncio
@aoverride_config
async def test_nl_to_answer_chat_async_chain_should_pass(
    mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI, fake_llm: FakeOpenAI, heavyiq_config
):
    nl_sql_chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db)
    chain = NLtoAnswerChain(llm=fake_llm, nl_sql_chain=nl_sql_chain, database=mock_heavy_db)

    chain_input = {
        chain.input_key: "What is the total population in the USA according to the data in the usa_states table?",
        "tables": ["usa_states"],
    }
    res = await log_chain_call_async(chain, chain_input, "")
    chain_output = res[chain.output_key]
    chain_sql_output = res[chain.output_sql_key]
    assert chain_output == "Total population in the USA according to the data in the usa_states table is 327514334"
    assert chain_sql_output == "SELECT SUM(POPULATION) AS total_population\nFROM usa_states;"
