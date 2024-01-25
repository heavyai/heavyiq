import time
from enum import Enum
from functools import lru_cache
from typing import Any

import requests
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import AzureOpenAI
from langchain.llms.base import BaseLLM

from heavyiq.config import get_config
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.utils import SharedDictSingleton

from .overrides import OverrideOpenAI, OverrideVLLMOpenAI


class LLMType(Enum):
    DEFAULT = "default"
    NL_TO_SQL = "nl_to_sql"
    SQL_TO_ANSWER = "sql_to_answer"
    NL_TO_TABLES = "nl_to_tables"
    INSTRUCT = "instruct"  # mainly used on /call-llm endpoint to resolve general instructions


def is_using_custom_trained_llm() -> bool:
    config = get_config()
    return config.custom_llm_type in ["API", "API_VLLM"]


def get_vllm_model_name(api_base: str) -> str:
    """
    Get VLLM model name either from cache or from remote endpoint.
    """
    logger = get_heavyiq_logger()
    shared_dict_instance: SharedDictSingleton | None = SharedDictSingleton._instance
    model_name_key, expiry_key, ttl = "VLLM_MODEL_NAME", "VLLM_MODEL_NAME_EXPIRES_AT", 60
    if (
        shared_dict_instance
        and (vllm_model_name_expires_at := shared_dict_instance.sget(expiry_key))
        and (vllm_model_name := shared_dict_instance.sget(model_name_key))
    ):
        # check for the time expired or not
        current_time = int(time.time())
        if current_time < vllm_model_name_expires_at:
            # return the model name from cache
            logger.debug("Getting VLLM model name from cache.")
            return vllm_model_name

    logger.debug("Getting VLLM model name.")
    response = requests.get(f"{api_base}/models", timeout=10)
    response.raise_for_status()
    model_name = response.json()["data"][0]["id"]
    if shared_dict_instance:
        shared_dict_instance.sput(expiry_key, int(time.time()) + ttl)
        shared_dict_instance.sput(model_name_key, model_name)

    return model_name


@lru_cache
def get_vllm_model_kwargs(model_type: LLMType) -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_config()
    # by default n was set to 1, so no need for passing n as llm kwargs otherwise if we need to
    kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if config.enable_logprobs and model_type == LLMType.NL_TO_SQL:
        model_kwargs["logprobs"] = config.custom_llm_logprobs_limit
    if config.custom_llm_api_vllm_beam_width >= 2 and model_type == LLMType.NL_TO_SQL:
        model_kwargs["use_beam_search"] = True
        kwargs["best_of"] = config.custom_llm_api_vllm_beam_width
        kwargs["n"] = 1
    if model_type == LLMType.NL_TO_SQL:
        kwargs["max_tokens"] = config.custom_llm_api_vllm_max_tokens
    return kwargs, model_kwargs


def get_llm_by_type(model_type: LLMType, **kwargs) -> BaseLLM | BaseChatModel:
    """
    Gets the relevant instantiated LLM class by llm type.
    Make sure to run this func inside run_in_threapool when called from a coroutine
    since it makes a network call to identify vllm.model_name.
    """
    config = get_config()
    if config.custom_llm_type is None or config.custom_llm_type == "AZURE":
        openai_llm_mapping: dict[LLMType, str | None] = {
            LLMType.DEFAULT: config.openai_gpt_model,
            LLMType.NL_TO_SQL: config.openai_gpt_model_nl_to_sql,
            LLMType.SQL_TO_ANSWER: config.openai_gpt_model_sql_to_answer,
            LLMType.NL_TO_TABLES: config.openai_gpt_model_nl_to_tables,
            LLMType.INSTRUCT: config.openai_gpt_model_instruct,
        }
        model_name: str = openai_llm_mapping[LLMType.DEFAULT] if openai_llm_mapping[model_type] is None else openai_llm_mapping[model_type]  # type: ignore
        return get_openai_llm_by_model_name(model=model_name, **kwargs)
    else:
        custom_llm_mapping: dict[LLMType, tuple[str | None, int]] = {
            LLMType.DEFAULT: (config.custom_llm_api_base, config.custom_llm_api_context_window),
            LLMType.NL_TO_SQL: (config.custom_llm_api_nl_to_sql_base, config.custom_llm_api_nl_to_sql_context_window),
            LLMType.SQL_TO_ANSWER: (
                config.custom_llm_api_sql_to_answer_base,
                config.custom_llm_api_sql_to_answer_context_window,
            ),
            LLMType.NL_TO_TABLES: (
                config.custom_llm_api_nl_to_tables_base,
                config.custom_llm_api_nl_to_tables_context_window,
            ),
            LLMType.INSTRUCT: (config.custom_llm_api_instruct_base, config.custom_llm_api_instruct_context_window),
        }
        api_base, context_window = (
            custom_llm_mapping[LLMType.DEFAULT]
            if custom_llm_mapping[model_type][0] is None
            else custom_llm_mapping[model_type]
        )
        return _get_custom_llm(model_type, api_base, context_window, **kwargs)  # type: ignore


def _get_custom_api_llm(model_type: LLMType, api_base: str, context_window: int, **kwargs):
    """
    Return the corresponding LLM class for the custom llm type API.(ie. llama2)
    """
    return OverrideOpenAI(
        openai_api_key="nothing",
        openai_api_base=api_base,
        model=f"CUSTOM_LLM_{model_type.value}",
        context_window=context_window,
        **kwargs,
    )


def _get_custom_api_vllm_llm(model_type: LLMType, api_base: str, context_window: int, **kwargs):
    """
    Return the corresponding LLM class for the custom llm type API_VLLM.(ie. VLLM)
    """
    # always pass immutable kwargs to the function decorated by lru_cache or otherwise you'll end up in
    # unhashable type list (ie. mutable) when passing a mutable object.
    vllm_kwargs, model_kwargs = get_vllm_model_kwargs(model_type)
    model_name = get_vllm_model_name(api_base)
    return OverrideVLLMOpenAI(
        openai_api_key="nothing",
        openai_api_base=api_base,
        model=model_name,
        model_kwargs=model_kwargs,
        context_window=context_window,
        **kwargs,
        **vllm_kwargs,
    )


def _get_custom_llm(model_type: LLMType, api_base: str, context_window: int, **kwargs) -> BaseLLM:
    config = get_config()
    if config.custom_llm_type == "API":
        return _get_custom_api_llm(model_type=model_type, api_base=api_base, context_window=context_window, **kwargs)
    elif config.custom_llm_type == "API_VLLM":
        return _get_custom_api_vllm_llm(
            model_type=model_type, api_base=api_base, context_window=context_window, **kwargs
        )
    else:
        raise ValueError(f"Invalid custom LLM type: {config.custom_llm_type}")


def get_openai_llm_by_model_name(model: str, **kwargs) -> BaseLLM | BaseChatModel:
    if model.startswith(("gpt-3.5", "gpt-4", "gpt-35")):
        return _get_openai_chat_llm(model, **kwargs)
    return _get_openai_llm(model, **kwargs)


def _get_openai_llm(model: str, **kwargs) -> BaseLLM:
    config = get_config()
    if config.custom_llm_type == "AZURE":
        return AzureOpenAI(
            openai_api_base=config.custom_llm_azure_openai_api_base,
            openai_api_key=config.openai_api_key,
            openai_api_version=config.custom_llm_azure_openai_api_version,
            deployment_name=config.custom_llm_azure_deployment_name,
            tiktoken_model_name=azure_model_to_openai(model),
            model=model,
            **kwargs,
        )
    return OverrideOpenAI(model=model, openai_api_key=config.openai_api_key, **kwargs)


def _get_openai_chat_llm(model: str, **kwargs) -> BaseChatModel:
    config = get_config()
    if config.custom_llm_type == "AZURE":
        return AzureChatOpenAI(
            openai_api_base=config.custom_llm_azure_openai_api_base,
            openai_api_key=config.openai_api_key,
            openai_api_version=config.custom_llm_azure_openai_api_version,
            deployment_name=config.custom_llm_azure_deployment_name,
            tiktoken_model_name=azure_model_to_openai(model),
            model=model,
            **kwargs,
        )

    return ChatOpenAI(model=model, openai_api_key=config.openai_api_key, **kwargs)


def get_llm(**kwargs) -> BaseLLM | BaseChatModel:
    return get_llm_by_type(LLMType.DEFAULT, **kwargs)


def azure_model_to_openai(model: str) -> str:
    if model.startswith("gpt-35"):
        return model.replace("gpt-35", "gpt-3.5")
    return model


def get_chat_llm(model: str, **kwargs) -> BaseChatModel:
    config = get_config()
    if config.custom_llm_type is not None:
        if config.custom_llm_type == "AZURE":
            return AzureChatOpenAI(
                openai_api_base=config.custom_llm_azure_openai_api_base,
                openai_api_key=config.openai_api_key,
                openai_api_version=config.custom_llm_azure_openai_api_version,
                deployment_name=config.custom_llm_azure_deployment_name,
                tiktoken_model_name=azure_model_to_openai(model),
                model=model,
                **kwargs,
            )
        else:
            raise NotImplementedError("Custom LLMs are not supported for chat models yet.")
    return ChatOpenAI(model=model, openai_api_key=config.openai_api_key, **kwargs)
