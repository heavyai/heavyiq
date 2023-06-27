from langchain.llms.openai import OpenAI
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI

from heavynl.config import get_config
from heavynl.langchain.utils import is_promptlayer_active


def get_llm_by_model_name(model: str, tags: list[str] = [], **kwargs) -> OpenAI | ChatOpenAI:
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        return get_chat_llm(tags=tags, model=model, **kwargs)
    else:
        return get_llm(tags=tags, model=model, **kwargs)


def get_llm(tags: list[str] = [], **kwargs) -> OpenAI:
    if is_promptlayer_active:
        return PromptLayerOpenAI(pl_tags=tags, return_pl_id=True, openai_api_key=get_config().openai_api_key, **kwargs)
    else:
        return OpenAI(openai_api_key=get_config().openai_api_key, **kwargs)


def get_chat_llm(tags: list[str] = [], **kwargs) -> ChatOpenAI:
    if is_promptlayer_active:
        return PromptLayerChatOpenAI(
            pl_tags=tags, return_pl_id=True, openai_api_key=get_config().openai_api_key, **kwargs
        )
    else:
        return ChatOpenAI(openai_api_key=get_config().openai_api_key, **kwargs)
