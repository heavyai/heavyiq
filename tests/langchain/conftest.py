# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from glob import glob
import pytest
from confz import DataSource

from heavyiq.config import get_config
from heavyiq.config.config_schema import HeavyIQConfig
from heavyiq.langchain import HeavyDB
from tests.langchain import FakeChatOpenAI, FakeOpenAI


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
