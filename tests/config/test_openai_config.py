# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from heavyiq.config import HeavyIQConfig, OpenAI, get_config


@pytest.fixture
def openai_mock():
    with patch("heavyiq.config.OpenAI") as mock_openai:
        mock_instance = mock_openai.return_value
        yield mock_instance


@pytest.mark.custom_config("openai_config")
# @patch("heavyiq.config.openai.Model.list", side_effect=ValueError("Connection Error"))
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_openai_llm_type_if_failed_to_communicate_with_openai_server(
    # MockClient: Any,
    conf_file: str,
    openai_mock,
):
    # here we mock the global _config variable to return None
    # why because being set it as a global variable, there might be a chances of the varibale
    # getting populated by the previous testcases

    openai_mock.models.list.side_effect = Exception("Connection Error")

    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)

    assert str(exc_info.value) == "Unable to communicate with OpenAI API: Connection Error"


@pytest.mark.custom_config("openai_config")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_pass_on_openai_llm_type_if_relevant_config_is_set(conf_file: str, openai_mock):
    openai_mock.models.list.return_value = []
    config: HeavyIQConfig = get_config(conf_file)
    assert config.openai_api_key == "some-dummy-key"


@pytest.mark.custom_config("openai_config")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_create_default_data_dir_if_not_exists(conf_file: str, openai_mock):
    openai_mock.models.list.return_value = []
    config: HeavyIQConfig = get_config(conf_file)
    assert config.data == "storage"
    assert os.path.exists(config.data)


@pytest.mark.custom_config("openai_config_with_data_dir")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_create_data_dir_if_not_exists(conf_file: str, openai_mock):
    openai_mock.models.list.return_value = []
    config: HeavyIQConfig = get_config(conf_file)
    assert config.data == "test-data-dir"
    assert os.path.exists(config.data)
    os.rmdir(config.data)
