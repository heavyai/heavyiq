# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch
from heavyiq.config import get_config, HeavyIQConfig


@pytest.mark.custom_config("custom_api_config_without_base")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_custom_api_llm_type_if_no_relevant_config_is_set(conf_file: str):
    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)
    assert str(exc_info.value) == "Custom LLM type is set to 'API', but API base URL is not set."


@pytest.mark.custom_config("custom_api_config_with_base")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_pass_on_custom_api_llm_type_if_relevant_config_is_set(conf_file: str):
    config: HeavyIQConfig = get_config(conf_file)
    assert config.custom_llm_type == "API"
    assert config.custom_llm_api_base == "http://localhost:9000/api/v1"


@pytest.mark.custom_config("custom_api_vllm_config_without_base")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_custom_api_vllm_llm_type_if_no_relevant_config_is_set(conf_file: str):
    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)
    assert str(exc_info.value) == "Custom LLM type is set to 'API_VLLM', but API base URL is not set."


@pytest.mark.custom_config("custom_api_vllm_config_with_base")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_pass_on_custom_api_vllm_llm_type_if_relevant_config_is_set(conf_file: str):
    config: HeavyIQConfig = get_config(conf_file)
    assert config.custom_llm_type == "API_VLLM"
    assert config.custom_llm_api_base == "http://localhost:9000/api/v1"


@pytest.mark.custom_config("custom_api_invalid")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_invalid_custom_api(conf_file: str):
    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)
    assert str(exc_info.value) == "Invalid custom LLM type (valid options are AZURE or API or API_VLLM): INVALID"
