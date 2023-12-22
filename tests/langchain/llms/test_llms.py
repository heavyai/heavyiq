from unittest.mock import patch

import pytest
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.llms import AzureOpenAI

from heavyiq.config import HeavyIQConfig
from heavyiq.langchain.llms import LLMType, get_llm_by_type, get_vllm_model_kwargs, is_using_custom_trained_llm
from heavyiq.langchain.llms.overrides import OverrideOpenAI, OverrideVLLMOpenAI


def test_should_check_whether_custom_llm_used_or_not():
    with patch(
        "heavyiq.langchain.llms.get_config", return_value=HeavyIQConfig(openai_api_key="", custom_llm_type="API")
    ):
        assert is_using_custom_trained_llm()
    with patch(
        "heavyiq.langchain.llms.get_config", return_value=HeavyIQConfig(openai_api_key="", custom_llm_type="API_VLLM")
    ):
        assert is_using_custom_trained_llm()
    with patch("heavyiq.langchain.llms.get_config", return_value=HeavyIQConfig(openai_api_key="", custom_llm_type="")):
        assert is_using_custom_trained_llm() == False


@pytest.mark.parametrize(
    "model_type, beam_width, expected",
    [
        (LLMType.DEFAULT, 1, ({}, {})),
        (LLMType.NL_TO_SQL, 2, ({"n": 1, "best_of": 2}, {"use_beam_search": True, "logprobs": 5})),
        (LLMType.SQL_TO_ANSWER, 2, ({}, {})),
    ],
)
def test_get_vllm_model_kwargs_should_return_model_kwargs_for_valid_llm_type(
    model_type,
    beam_width,
    expected,
):
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="",
            custom_llm_type="API_VLLM",
            custom_llm_api_vllm_beam_width=beam_width,
            enable_logprobs=True,
            custom_llm_logprobs_limit=5,
        ),
    ):
        kwargs, model_kwargs = get_vllm_model_kwargs(model_type)
        assert kwargs == expected[0]
        assert model_kwargs == expected[1]


@pytest.mark.parametrize(
    "model_type, models, expected",
    [
        (LLMType.DEFAULT, ("gpt-4", None, None), (ChatOpenAI, "gpt-4")),
        (LLMType.NL_TO_SQL, ("text-davinci-003", None, None), (OverrideOpenAI, "text-davinci-003")),
        (LLMType.NL_TO_SQL, ("text-davinci-003", "gpt-3.5", None), (ChatOpenAI, "gpt-3.5")),
        (LLMType.SQL_TO_ANSWER, ("gpt-4", None, "gpt-3.5"), (ChatOpenAI, "gpt-3.5")),
        (LLMType.SQL_TO_ANSWER, ("gpt-4", None, None), (ChatOpenAI, "gpt-4")),
    ],
)
def test_llm_by_type_should_return_corresponding_openai_llm(model_type, models, expected):
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="dummy-key",
            openai_gpt_model=models[0],
            openai_gpt_model_nl_to_sql=models[1],
            openai_gpt_model_sql_to_answer=models[2],
        ),
    ):
        llm = get_llm_by_type(model_type=model_type)
        assert isinstance(llm, expected[0])
        assert llm.model_name == expected[1]


@pytest.mark.parametrize(
    "model_type, models, expected",
    [
        (LLMType.DEFAULT, ("gpt-4", None, None), (AzureChatOpenAI, "gpt-4")),
        (LLMType.NL_TO_SQL, ("text-davinci-003", None, None), (AzureOpenAI, "text-davinci-003")),
        (LLMType.NL_TO_SQL, ("text-davinci-003", "gpt-3.5", None), (AzureChatOpenAI, "gpt-3.5")),
        (LLMType.SQL_TO_ANSWER, ("gpt-4", None, "gpt-3.5"), (AzureChatOpenAI, "gpt-3.5")),
        (LLMType.SQL_TO_ANSWER, ("gpt-4", None, None), (AzureChatOpenAI, "gpt-4")),
    ],
)
def test_llm_by_type_should_return_corresponding_custom_azure_llm(model_type, models, expected):
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="dummy-key",
            custom_llm_type="AZURE",
            custom_llm_azure_openai_api_base="http://localhost:7777",
            openai_gpt_model=models[0],
            openai_gpt_model_nl_to_sql=models[1],
            openai_gpt_model_sql_to_answer=models[2],
        ),
    ):
        llm = get_llm_by_type(model_type=model_type)
        assert isinstance(llm, expected[0])
        assert llm.model_name == expected[1]


@pytest.mark.parametrize(
    "model_type, custom_params, expected",
    [
        (
            LLMType.DEFAULT,
            ("http://localhost:4000", 2048, None, 0, None, 0, None, 0),
            (OverrideOpenAI, "http://localhost:4000", "CUSTOM_LLM_default", 2048),
        ),
        (
            LLMType.NL_TO_SQL,
            ("http://localhost:5000", 4096, "http://localhost:6000", 2048, None, 0, None, 0),
            (OverrideOpenAI, "http://localhost:6000", "CUSTOM_LLM_nl_to_sql", 2048),
        ),
        (
            LLMType.NL_TO_SQL,
            ("http://localhost:4000", 2048, None, 0, None, 0, None, 0),
            (OverrideOpenAI, "http://localhost:4000", "CUSTOM_LLM_nl_to_sql", 2048),
        ),
        (
            LLMType.SQL_TO_ANSWER,
            ("http://localhost:4000", 2048, None, 0, "http://localhost:8000", 2048, None, 0),
            (OverrideOpenAI, "http://localhost:8000", "CUSTOM_LLM_sql_to_answer", 2048),
        ),
        (
            LLMType.SQL_TO_ANSWER,
            ("http://localhost:4000", 4096, None, 0, None, 0, None, 0),
            (OverrideOpenAI, "http://localhost:4000", "CUSTOM_LLM_sql_to_answer", 4096),
        ),
        (
            LLMType.NL_TO_TABLES,
            ("http://localhost:4000", 4096, None, 0, None, 0, "http://localhost:5000", 8192),
            (OverrideOpenAI, "http://localhost:5000", "CUSTOM_LLM_nl_to_tables", 8192),
        ),
        (
            LLMType.NL_TO_TABLES,
            (
                "http://localhost:4000",
                4096,
                None,
                0,
                None,
                0,
                None,
                8192,
            ),  # if without nl-ot-tables base, then it would take default base and default context window ir-respective of the appropriate context width
            (OverrideOpenAI, "http://localhost:4000", "CUSTOM_LLM_nl_to_tables", 4096),
        ),
    ],
)
def test_llm_by_type_should_return_overrided_openai_llm_for_api_custom_type(model_type, custom_params, expected):
    (
        custom_llm_api_base,
        custom_llm_api_context_window,
        custom_llm_api_nl_to_sql_base,
        custom_llm_api_nl_to_sql_context_window,
        custom_llm_api_sql_to_answer_base,
        custom_llm_api_sql_to_answer_context_window,
        custom_llm_api_nl_to_tables_base,
        custom_llm_api_nl_to_tables_context_window,
    ) = custom_params
    expected_llm, expected_base, expected_model_name, expected_context = expected
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="dummy-key",
            custom_llm_type="API",
            custom_llm_api_base=custom_llm_api_base,
            custom_llm_api_context_window=custom_llm_api_context_window,
            custom_llm_api_nl_to_sql_base=custom_llm_api_nl_to_sql_base,
            custom_llm_api_nl_to_sql_context_window=custom_llm_api_nl_to_sql_context_window,
            custom_llm_api_sql_to_answer_base=custom_llm_api_sql_to_answer_base,
            custom_llm_api_sql_to_answer_context_window=custom_llm_api_sql_to_answer_context_window,
            custom_llm_api_nl_to_tables_base=custom_llm_api_nl_to_tables_base,
            custom_llm_api_nl_to_tables_context_window=custom_llm_api_nl_to_tables_context_window,
        ),
    ):
        llm = get_llm_by_type(model_type=model_type)
        assert isinstance(llm, expected_llm)
        assert llm.openai_api_base == expected_base
        assert llm.model_name == expected_model_name
        assert llm.context_window == expected_context


@patch("heavyiq.langchain.llms.get_vllm_model_name", return_value="dummy-model")
def test_llm_by_type_should_return_oevrrided_vllm_for_vllm_api_custom_type(mock):
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="dummy-key",
            custom_llm_type="API_VLLM",
            custom_llm_api_base="http://localhost:4000",
            custom_llm_api_context_window=4096,
        ),
    ):
        llm = get_llm_by_type(model_type=LLMType.DEFAULT)
        assert isinstance(llm, OverrideVLLMOpenAI)
        assert llm.openai_api_base == "http://localhost:4000"
        assert llm.model_name == "dummy-model"
        assert llm.context_window == 4096
