import pytest
from unittest.mock import patch
from heavyiq.config import HeavyIQConfig
from heavyiq.langchain.llms import is_using_custom_trained_llm, LLMType, get_vllm_model_kwargs


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
    "model_type, beam_width, method_kwargs, expected",
    [
        (LLMType.DEFAULT, 1, {"n": 1}, ({"n": 1}, {})),
        (LLMType.NL_TO_SQL, 2, {"n": 2}, ({"n": 2, "best_of": 2}, {"use_beam_search": True})),
        (LLMType.SQL_TO_ANSWER, 2, {"n": 1}, ({"n": 1}, {})),
    ],
)
def test_get_vllm_model_kwargs_should_return_model_kwargs_for_valid_llm_type(
    model_type, beam_width, method_kwargs, expected
):
    with patch(
        "heavyiq.langchain.llms.get_config",
        return_value=HeavyIQConfig(
            openai_api_key="", custom_llm_type="API_VLLM", custom_llm_api_vllm_beam_width=beam_width
        ),
    ):
        kwargs, model_kwargs = get_vllm_model_kwargs(model_type, **method_kwargs)
        assert kwargs == expected[0]
        assert model_kwargs == expected[1]
