from unittest.mock import patch

import pytest
from confz import DataSource

from heavyiq.config import get_config
from heavyiq.config.config_schema import HeavyIQConfig
from heavyiq.langchain import HeavyDB
from tests.langchain import FakeChatOpenAI, FakeOpenAI


@pytest.fixture(scope="package")
def mock_heavy_db():
    new_source = DataSource(
        data={
            "heavydb_dbname": "heavynl",
            "heavydb_username": "admin",
            "heavydb_password": "HyperInteractive",
            "heavydb_host": "10.2.1.33",
            "heavydb_port": "6274",
            "heavydb_protocol": "binary",
            "openai_api_key": "",
        }
    )
    try:
        # Use patch to mock the get_config function
        with HeavyIQConfig.change_config_sources(new_source):
            with patch("heavyiq.langchain.heavydb.get_config", return_value=HeavyIQConfig()):
                yield HeavyDB.from_env()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None

    print("Tearing heavydb mock client")


@pytest.fixture(scope="package")
def fake_chat_llm():
    config: HeavyIQConfig = get_config()
    yield FakeChatOpenAI(
        model=config.openai_gpt_model_nl_to_sql or config.openai_gpt_model,
        temperature=0.0,
        openai_api_key=config.openai_api_key,
    )


@pytest.fixture(scope="package")
def fake_llm():
    config: HeavyIQConfig = get_config()
    yield FakeOpenAI(
        model=config.openai_gpt_model_sql_to_answer or config.openai_gpt_model,
        temperature=0.0,
        openai_api_key=config.openai_api_key,
    )
