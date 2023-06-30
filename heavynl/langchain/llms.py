from langchain.llms import AzureOpenAI, OpenAI
from langchain.llms.base import BaseLLM
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI, AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel

from heavynl.config import get_config
from heavynl.langchain.utils import is_promptlayer_active


def get_llm_by_model_name(model: str, tags: list[str] = [], **kwargs) -> BaseLLM | BaseChatModel:
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        return get_chat_llm(tags=tags, model=model, **kwargs)
    else:
        return get_llm(tags=tags, model=model, **kwargs)


def get_llm(tags: list[str] = [], **kwargs) -> BaseLLM:
    config = get_config()
    if config.custom_llm is not None:
        custom = config.custom_llm
        if custom.type == "AZURE":
            return AzureOpenAI(
                openai_api_base=config.custom_llm.openai_api_base,
                openai_api_key=config.openai_api_key,
                openai_api_version=config.custom_llm.openai_api_version,
                deployment_name=custom.azure_deployment_name,
                **kwargs
            )
        elif custom.type == "API":
            if is_promptlayer_active:
                return PromptLayerOpenAI(
                    pl_tags=tags,
                    return_pl_id=True,
                    openai_api_key="not_necessary",
                    openai_api_base=custom.api_base,
                    **kwargs
                )
            else:
                return OpenAI(openai_api_key="nothing", openai_api_base=custom.api_base, **kwargs)
    if is_promptlayer_active:
        return PromptLayerOpenAI(pl_tags=tags, return_pl_id=True, openai_api_key=config.openai_api_key, **kwargs)
    else:
        return OpenAI(openai_api_key=config.openai_api_key, **kwargs)


def get_chat_llm(tags: list[str] = [], **kwargs) -> ChatOpenAI:
    config = get_config()
    if config.custom_llm is not None:
        custom = config.custom_llm
        if custom.type == "AZURE":
            return AzureChatOpenAI(
                openai_api_base=config.custom_llm.openai_api_base,
                openai_api_key=config.openai_api_key,
                openai_api_version=config.custom_llm.openai_api_version,
                deployment_name=custom.azure_deployment_name,
                tiktoken_model_name=config.openai_gpt_model if not config.custom_llm else "gpt-3.5-turbo",
                **kwargs
            )
        else:
            raise NotImplementedError("Custom LLMs are not supported for chat models yet.")
    if is_promptlayer_active:
        return PromptLayerChatOpenAI(pl_tags=tags, return_pl_id=True, openai_api_key=config.openai_api_key, **kwargs)
    else:
        return ChatOpenAI(openai_api_key=config.openai_api_key, **kwargs)
