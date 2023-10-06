import re
import pytest
import aiofiles
from heavyiq.logging_utils import HeavyIQLogger, get_config
from heavyiq.langchain.callbacks import AsyncFileCallbackHandler, AsyncLogFileCallbackHandler
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.logging import log_chain_call_async
from heavyiq.langchain.chains import NLtoSQLChatChain
from tests.langchain import FakeChatOpenAI


@pytest.mark.asyncio
async def test_async_file_callback_handler_should_pass(
    log_file: str, mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI
):
    callback_handler = AsyncFileCallbackHandler(log_file, to_stdout=False)
    # question/chain_input wouldn't be considered since we are use FakeLLM
    question = "What is the total population in the USA according to the data in the usa_states table?"

    chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db, callbacks=[callback_handler])
    chain_input = {chain.input_key: question, "tables": ["usa_states"]}
    res = await log_chain_call_async(chain, chain_input, "")
    chain_output = res[chain.output_key]
    assert chain_output

    chain_output = chain_output.rstrip(";")

    # check for chain output logged on the target file
    async with aiofiles.open(callback_handler.file_path, mode="r") as file:
        contents = await file.read()
        assert chain_output in contents


def get_mock_heavyiq_logger(log_file_path: str) -> HeavyIQLogger:
    LOG_CONFIG = get_config()
    return HeavyIQLogger(
        log_file_path=log_file_path,
        level=LOG_CONFIG.heavyiq_log_level,
        enable_console_logging=LOG_CONFIG.log_to_stdout,
    )


@pytest.mark.asyncio
async def test_async_log_file_callback_handler_should_pass(
    log_file: str, mock_heavy_db: HeavyDB, fake_chat_llm: FakeChatOpenAI
):
    callback_handler = AsyncLogFileCallbackHandler(log_file, to_stdout=False)
    callback_handler._logger = get_mock_heavyiq_logger(log_file)

    question = "What is the total population in the USA according to the data in the usa_states table?"

    chain = NLtoSQLChatChain(llm=fake_chat_llm, database=mock_heavy_db, callbacks=[callback_handler])
    chain_input = {chain.input_key: question, "tables": ["usa_states"]}
    res = await log_chain_call_async(chain, chain_input, "")
    chain_output = res[chain.output_key]
    assert chain_output

    chain_output = chain_output.rstrip(";")

    # check for chain output logged on the target file
    async with aiofiles.open(log_file, mode="r") as file:
        contents = await file.read()
        assert chain_output in contents
