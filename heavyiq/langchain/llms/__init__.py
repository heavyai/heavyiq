import time
from enum import Enum
from typing import Any

import httpx
import requests
from cachetools import LRUCache, TTLCache, cached
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain_openai.llms import AzureOpenAI

from heavyiq.config import get_config
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.utils import SharedDictSingleton

from .overrides import OverrideOpenAI, OverrideVLLMOpenAI


class LLMType(Enum):
    DEFAULT = "default"
    NL_TO_SQL = "nl_to_sql"
    NL_TO_SQL_COT = "nl_to_sql_cot"
    NL_TO_MULTIPLE_SQL = "nl_to_multiple_sql"
    NL_TO_MULTIPLE_SQL_JUDGE = "nl_to_multiple_sql_judge"
    NL_TO_SQL_GEN = "nl_to_sql_gen"  # nl to sql gen model used to multiple sql queries
    NL_TO_SQL_ERROR = "nl_to_sql_error"
    NL_TO_SQL_COT_ERROR = "nl_to_sql_cot_error"
    SQL_TO_ANSWER = "sql_to_answer"
    NL_TO_TABLES = "nl_to_tables"
    TABLES_TO_QUESTIONS = "tables_to_questions"
    INSTRUCT = "instruct"  # mainly used on /call-llm endpoint to resolve general instructions
    RAG = "rag"


def is_using_custom_trained_llm() -> bool:
    config = get_config()
    return config.custom_llm_type in ["API", "API_VLLM"]


@cached(
    cache=TTLCache(maxsize=3, ttl=60 * 10)
)  # a max of 3 unique calls can be cached at a time and each can live for 10 minutes
def get_vllm_model_name(api_base: str) -> str:
    """
    Get VLLM model name either from cache or from remote endpoint.
    """
    logger, config = get_heavyiq_logger(), get_config()
    headers = {"Authorization": f"Bearer {config.heavylm_api_key}"} if config.heavylm_api_key else None
    logger.debug("Getting VLLM model name.")
    response = requests.get(f"{api_base}/models", headers=headers, timeout=10)
    response.raise_for_status()
    model_name = response.json()["data"][0]["id"]

    return model_name


def get_vllm_max_model_len(api_base: str) -> int:
    """
    Fetch the max_model_len from the embedding model listed by the vLLM server.

    Args:
        api_base (str): Base URL of the vLLM server (e.g., "http://localhost:8000/v1")

    Returns:
        int: max_model_len of the first model
    """
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{api_base}/models")
        resp.raise_for_status()
        data = resp.json()

        if not data.get("data"):
            raise ValueError("No models found in /v1/models response")

        first_model = data["data"][0]
        if "max_model_len" in first_model:
            return first_model["max_model_len"]

        raise ValueError("max_model_len not found in model metadata")


@cached(cache=LRUCache(maxsize=10))
def get_vllm_model_kwargs(model_type: LLMType) -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_config()
    # by default n was set to 1, so no need for passing n as llm kwargs otherwise if we need to
    kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    # set seed for all model type
    seed_value = 42
    kwargs["seed"] = seed_value
    if config.enable_logprobs and model_type in [LLMType.NL_TO_SQL, LLMType.NL_TO_SQL_ERROR]:
        model_kwargs["logprobs"] = config.custom_llm_logprobs_limit
    if config.custom_llm_api_vllm_beam_width >= 2 and model_type in [LLMType.NL_TO_SQL, LLMType.NL_TO_SQL_ERROR]:
        model_kwargs["use_beam_search"] = True
        kwargs["best_of"] = config.custom_llm_api_vllm_beam_width
        kwargs["n"] = 1
    if model_type == LLMType.NL_TO_SQL_GEN:
        kwargs["max_tokens"] = config.custom_llm_api_vllm_max_tokens
        # set best_of >= n for sql generations
        kwargs["best_of"] = 5
        model_kwargs["use_beam_search"] = False  # if we set beam search then temperature should not be 0
        kwargs["n"] = 5
    if model_type in [LLMType.NL_TO_SQL, LLMType.NL_TO_SQL_GEN, LLMType.NL_TO_SQL_ERROR]:
        kwargs["max_tokens"] = config.custom_llm_api_vllm_max_tokens
    if model_type in [LLMType.NL_TO_MULTIPLE_SQL]:
        kwargs["max_tokens"] = config.custom_llm_api_vllm_max_tokens
        model_kwargs["use_beam_search"] = False
        kwargs["n"] = 5
    if model_type == LLMType.NL_TO_MULTIPLE_SQL_JUDGE:
        # enable logprobs
        model_kwargs["logprobs"] = config.custom_llm_logprobs_limit
        kwargs["n"] = 1
        kwargs["max_tokens"] = config.custom_llm_api_vllm_max_tokens
    if model_type == LLMType.INSTRUCT:
        model_kwargs["use_beam_search"] = False
    return kwargs, model_kwargs


@cached(
    cache=TTLCache(maxsize=20, ttl=60 * 10)
)  # a max of 20 unique calls can be cached at a time and each can live for 10 minutes
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
            LLMType.NL_TO_SQL_ERROR: config.openai_gpt_model_nl_to_sql_error,
            LLMType.SQL_TO_ANSWER: config.openai_gpt_model_sql_to_answer,
            LLMType.NL_TO_TABLES: config.openai_gpt_model_nl_to_tables,
            LLMType.INSTRUCT: config.openai_gpt_model_instruct,
            LLMType.TABLES_TO_QUESTIONS: config.openai_gpt_model_tables_to_questions,
            LLMType.NL_TO_SQL_GEN: config.openai_gpt_model_nl_to_sql,  # use the same nl to sql model
            LLMType.RAG: config.openai_gpt_model,  # use the same default model for handling RAG question
        }
        model_name: str = openai_llm_mapping[LLMType.DEFAULT] if openai_llm_mapping[model_type] is None else openai_llm_mapping[model_type]  # type: ignore
        return get_openai_llm_by_model_name(model=model_name, **kwargs)
    else:
        custom_llm_mapping: dict[LLMType, tuple[str | None, int]] = {
            LLMType.DEFAULT: (config.custom_llm_api_base, config.custom_llm_api_context_window),
            LLMType.NL_TO_SQL: (config.custom_llm_api_nl_to_sql_base, config.custom_llm_api_nl_to_sql_context_window),
            LLMType.NL_TO_SQL_ERROR: (
                config.custom_llm_api_nl_to_sql_error_base,
                config.custom_llm_api_nl_to_sql_error_context_window,
            ),
            LLMType.SQL_TO_ANSWER: (
                config.custom_llm_api_sql_to_answer_base,
                config.custom_llm_api_sql_to_answer_context_window,
            ),
            LLMType.NL_TO_TABLES: (
                config.custom_llm_api_nl_to_tables_base,
                config.custom_llm_api_nl_to_tables_context_window,
            ),
            LLMType.INSTRUCT: (config.custom_llm_api_instruct_base, config.custom_llm_api_instruct_context_window),
            LLMType.TABLES_TO_QUESTIONS: (
                config.custom_llm_api_tables_to_questions_base,
                config.custom_llm_api_tables_to_questions_context_window,
            ),  # uses the default model and context window
            LLMType.NL_TO_SQL_GEN: (
                config.custom_llm_api_nl_to_sql_base,
                config.custom_llm_api_nl_to_sql_context_window,
            ),
            LLMType.RAG: (config.custom_llm_api_rag_base, config.custom_llm_api_rag_context_window),
        }
        if model_type == LLMType.NL_TO_SQL_ERROR:
            found_key = LLMType.DEFAULT
            for key in [LLMType.NL_TO_SQL_ERROR, LLMType.NL_TO_SQL]:
                if custom_llm_mapping[key][0]:
                    found_key = key
                    break
            api_base, context_window = custom_llm_mapping[found_key]
        else:
            if model_type not in custom_llm_mapping:
                api_base, context_window = custom_llm_mapping[LLMType.DEFAULT]
            else:
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
    vllm_kwargs, extra_body = get_vllm_model_kwargs(model_type)
    model_name = get_vllm_model_name(api_base)
    combined_kwargs = {**vllm_kwargs, **kwargs}
    return OverrideVLLMOpenAI(
        openai_api_key="nothing",
        openai_api_base=api_base,
        model=model_name,
        model_kwargs={"extra_body": extra_body},
        context_window=context_window,
        **combined_kwargs,
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
            azure_endpoint=config.custom_llm_azure_openai_api_base,
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
            azure_endpoint=config.custom_llm_azure_openai_api_base,
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
