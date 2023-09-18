import pytest
from unittest.mock import patch
from typing import Any

from heavyiq.config import get_config, HeavyIQConfig


@pytest.mark.custom_config("openai_config")
@patch("heavyiq.config.openai.Model.list", side_effect=ValueError("Connection Error"))
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_raise_error_on_openai_llm_type_if_failed_to_communicate_with_openai_server(
    MockClient: Any, conf_file: str
):
    # here we mock the global _config variable to return None
    # why because being set it as a global variable, there might be a chances of the varibale
    # getting populated by the previous testcases

    with pytest.raises(ValueError) as exc_info:
        get_config(conf_file)

    assert str(exc_info.value) == "Unable to communicate with OpenAI API: Connection Error"


@pytest.mark.custom_config("openai_config")
@patch("heavyiq.config.openai.Model.list")
@patch("heavyiq.config._config", new=None)  # type: ignore
def test_get_config_should_pass_on_openai_llm_type_if_relevant_config_is_set(MockClient: Any, conf_file: str):
    MockClient.return_value = []
    config: HeavyIQConfig = get_config(conf_file)
    assert config.openai_api_key == "some-dummy-key"
