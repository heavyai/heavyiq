from enum import Enum
import requests
from functools import lru_cache
from typing import Any

from langchain.llms import AzureOpenAI
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel

from heavyiq.config import get_config
from .overrides import OverrideOpenAI, OverrideVLLMOpenAI


class LLMType(Enum):
    ANY = "any"
    NL_TO_SQL = "nl_to_sql"
    SQL_TO_ANSWER = "sql_to_answer"


def get_vllm_model_name(api_base: str) -> str:
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    return response.json()["data"][0]["id"]


@lru_cache
def get_vllm_model_kwargs(model_type: LLMType, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_config()
    model_kwargs: dict[str, Any] = {}
    if config.custom_llm_api_vllm_beam_width >= 2 and model_type == LLMType.NL_TO_SQL:
        model_kwargs["use_beam_search"] = True
        kwargs["best_of"] = config.custom_llm_api_vllm_beam_width
    return kwargs, model_kwargs


def get_llm_by_type(model_type: LLMType, **kwargs) -> BaseLLM | BaseChatModel:
    config = get_config()
    if config.custom_llm_type in ["API", "API_VLLM"]:
        if config.dev:
            dev_attr = getattr(config.dev, model_type.value, None)
            if dev_attr:
                return get_custom_llm(model_type, **dev_attr.dict(), **kwargs)
            else:
                raise ValueError(f"Unknown LLM type: {model_type}")
        return get_llm(model_type, **kwargs)
    return get_llm_by_model_name(model=config.openai_gpt_model, model_type=model_type, **kwargs)


def get_custom_llm(model_type: LLMType, api_base: str, context_window: int, **kwargs) -> BaseLLM:
    config = get_config()
    if config.custom_llm_type == "API":
        return OverrideOpenAI(
            openai_api_key="nothing",
            openai_api_base=api_base,
            model=f"CUSTOM_LLM_{model_type.value}",
            context_window=context_window,
            **kwargs,
        )
    kwargs, model_kwargs = get_vllm_model_kwargs(model_type, **kwargs)
    model_name = get_vllm_model_name(api_base)
    return OverrideVLLMOpenAI(
        openai_api_key="nothing",
        openai_api_base=api_base,
        model=model_name,
        model_kwargs=model_kwargs,
        context_window=context_window,
        **kwargs,
    )


def get_llm_by_model_name(model: str, model_type: LLMType = LLMType.ANY, **kwargs) -> BaseLLM | BaseChatModel:
    config = get_config()
    if config.custom_llm_type in ["API", "API_VLLM"]:
        return get_llm(model_type, **kwargs)
    if model.startswith(("gpt-3.5", "gpt-4", "gpt-35")):
        return get_chat_llm(model=model, **kwargs)
    return get_llm(model_type, model=model, **kwargs)


def get_llm(model_type: LLMType, **kwargs) -> BaseLLM:
    config = get_config()
    if config.custom_llm_type == "AZURE":
        return AzureOpenAI(
            openai_api_base=config.custom_llm_azure_openai_api_base,
            openai_api_key=config.openai_api_key,
            openai_api_version=config.custom_llm_azure_openai_api_version,
            deployment_name=config.custom_llm_azure_deployment_name,
            **kwargs,
        )
    if config.custom_llm_type == "API":
        return OverrideOpenAI(
            openai_api_key="nothing",
            openai_api_base=config.custom_llm_api_base,
            model="CUSTOM_LLM",
            context_window=config.custom_llm_api_context_window,
            **kwargs,
        )
    elif config.custom_llm_type == "API_VLLM":
        kwargs, model_kwargs = get_vllm_model_kwargs(model_type, **kwargs)
        model_name = get_vllm_model_name(config.custom_llm_api_base)
        return OverrideVLLMOpenAI(
            openai_api_key="nothing",
            openai_api_base=config.custom_llm_api_base,
            model=model_name,
            model_kwargs=model_kwargs,
            context_window=config.custom_llm_api_context_window,
            **kwargs,
        )
    return OverrideOpenAI(openai_api_key=config.openai_api_key, **kwargs)


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
