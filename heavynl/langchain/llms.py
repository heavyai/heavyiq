import os

from langchain.llms.openai import OpenAI
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI

from heavynl.config import get_config
from heavynl.langchain.utils import is_promptlayer_active

os.environ["OPENAI_API_KEY"] = get_config().openai_api_key


def get_llm_by_model_name(model_name: str, tags: list[str] = [], **kwargs) -> OpenAI | ChatOpenAI:
    if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
        return get_chat_llm(tags=tags, model_name=model_name, **kwargs)
    else:
        return get_llm(tags=tags, model_name=model_name, **kwargs)


def get_llm(tags: list[str] = [], **kwargs) -> OpenAI:
    if is_promptlayer_active:
        return PromptLayerOpenAI(pl_tags=tags, return_pl_id=True, **kwargs)
    else:
        return OpenAI(**kwargs)


def get_chat_llm(tags: list[str] = [], **kwargs) -> ChatOpenAI:
    if is_promptlayer_active:
        return PromptLayerChatOpenAI(pl_tags=tags, return_pl_id=True, **kwargs)
    else:
        return ChatOpenAI(**kwargs)
