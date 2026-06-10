# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch
from heavyiq.config import get_config, HeavyIQConfig


@pytest.mark.custom_config("azure_config_without_options")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_azure_llm_type_if_no_relevant_config_is_set(conf_file: str):
    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)
    assert (
        str(exc_info.value)
        == "Custom LLM type is set to 'AZURE', but deployment name or API base URL or API Version is not set."
    )


@pytest.mark.custom_config("azure_config_with_options")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_pass_on_azure_llm_type_if_relevant_config_is_set(conf_file: str):
    config: HeavyIQConfig = get_config(conf_file)
    assert config.custom_llm_type == "AZURE"
    assert config.custom_llm_azure_deployment_name == "heavyai"
    assert config.custom_llm_azure_openai_api_base == "https://azure.heavyai/llm/api/v1"
    assert config.custom_llm_azure_openai_api_version == "0.1"
