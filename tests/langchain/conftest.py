from unittest.mock import patch
from glob import glob
import pytest
from confz import DataSource

from heavyiq.config import get_config
from heavyiq.config.config_schema import HeavyIQConfig
from heavyiq.langchain import HeavyDB
from tests.langchain import FakeChatOpenAI, FakeOpenAI


# @pytest.fixture(scope="package")
# def heavyiq_config(config_file_path):
#     """
#     Fixture that supposed to return HeavyIQConfig by reading the config from passed config_file_path fixture..
#     """
#     print(f"Getting config from {config_file_path}")
#     with patch("heavyiq.config._config", new=None):
#         yield get_config(config_file_path)


# Stayed here for reference
# @pytest.fixture(scope="package")
# def mock_heavy_db(heavyiq_config: HeavyIQConfig):
#     new_source = DataSource(
#         data={
#             "heavydb_dbname": "heavynl",
#             "heavydb_username": "admin",
#             "heavydb_password": "HyperInteractive",
#             "heavydb_host": "10.2.1.33",
#             "heavydb_port": "6274",
#             "heavydb_protocol": "binary",
#             "openai_api_key": "",
#         }
#     )
#     try:
#         # Use patch to mock the get_config function
#         with HeavyIQConfig.change_config_sources(new_source):
#             with patch("heavyiq.langchain.heavydb.get_config", return_value=HeavyIQConfig()):
#                 yield HeavyDB.from_env()
#     except Exception as e:
#         print(f"An error occurred during mock heavydb creation: {e}")
#         yield None


#     print("Tearing heavydb mock client")
@pytest.fixture(scope="package")
def mock_heavy_db(heavyiq_config: HeavyIQConfig):
    try:
        # Use patch to mock the get_config function
        with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
            yield HeavyDB.from_env()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None


@pytest.fixture(scope="package")
def fake_chat_llm(heavyiq_config: HeavyIQConfig):
    yield FakeChatOpenAI(
        model=heavyiq_config.openai_gpt_model_nl_to_sql or heavyiq_config.openai_gpt_model,
        temperature=0.0,
        openai_api_key=heavyiq_config.openai_api_key,
    )


@pytest.fixture(scope="package")
def fake_llm(heavyiq_config: HeavyIQConfig):
    yield FakeOpenAI(
        model=heavyiq_config.openai_gpt_model_sql_to_answer or heavyiq_config.openai_gpt_model,
        temperature=0.0,
        openai_api_key=heavyiq_config.openai_api_key,
    )
