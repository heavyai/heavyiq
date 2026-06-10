# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from pathlib import Path
from _pytest.fixtures import FixtureRequest
import logging


@pytest.fixture(scope="function")
def conf_file(tmp_path: Path, request: FixtureRequest):
    # Create a temporary directory to store the sample.conf file
    logging.info("Triggering conf file")
    temp_dir = tmp_path / "config_files"
    temp_dir.mkdir()

    # Define the content of the sample.conf file
    azure_config_without_options = """
    [iq]
    custom_llm_type = "AZURE"
    custom_llm_azure_deployment_name = ""
    custom_llm_azure_openai_api_base = ""
    custom_llm_azure_openai_api_version = "1.2"
    openai_api_key = ""
    """
    azure_config_with_options = """
    [iq]
    custom_llm_type = "AZURE"
    custom_llm_azure_deployment_name = "heavyai"
    custom_llm_azure_openai_api_base = "https://azure.heavyai/llm/api/v1"
    custom_llm_azure_openai_api_version = "0.1"
    openai_api_key = ""
    """
    openai_config = """
    [iq]
    openai_api_key = "some-dummy-key"
    custom_llm_type = ""
    """
    openai_config_with_data_dir = """
    data = "test-data-dir"
    [iq]
    openai_api_key = "some-dummy-key"
    custom_llm_type = ""
    """
    custom_api_config_without_base = """
    [iq]
    openai_api_key = ""
    custom_llm_type = "API"
    custom_llm_api_base = ""
    """
    custom_api_config_with_base = """
    [iq]
    openai_api_key = ""
    custom_llm_type = "API"
    custom_llm_api_base = "http://localhost:9000/api/v1"
    """
    custom_api_vllm_config_without_base = """
    [iq]
    openai_api_key = ""
    custom_llm_type = "API_VLLM"
    custom_llm_api_base = ""
    """
    custom_api_vllm_config_with_base = """
    [iq]
    openai_api_key = ""
    custom_llm_type = "API_VLLM"
    custom_llm_api_base = "http://localhost:9000/api/v1"
    """
    custom_api_invalid = """
    [iq]
    openai_api_key = ""
    custom_llm_type = "INVALID"
    """

    configs = {
        "azure_config_with_options": azure_config_with_options,
        "azure_config_without_options": azure_config_without_options,
        "openai_config": openai_config,
        "custom_api_config_without_base": custom_api_config_without_base,
        "custom_api_config_with_base": custom_api_config_with_base,
        "custom_api_vllm_config_without_base": custom_api_vllm_config_without_base,
        "custom_api_vllm_config_with_base": custom_api_vllm_config_with_base,
        "custom_api_invalid": custom_api_invalid,
        "openai_config_with_data_dir": openai_config_with_data_dir,
    }

    # Get the custom value specified in the test function
    custom_config = request.node.get_closest_marker("custom_config")
    config_content = ""
    if custom_config:
        config_name = custom_config.args[0]  # type: ignore
        config_content = configs.get(config_name, "")  # Return the specified configuration

    # Create the config.conf file in the temporary directory
    conf_file_path = temp_dir / "config.conf"
    conf_file_path.write_text(config_content)
    # Return the path to the config.conf file
    yield str(conf_file_path)

    os.remove(str(conf_file_path))
