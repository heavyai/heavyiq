import os

from langchain.llms.openai import OpenAI
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI

from modules.config import config
from modules.langchain.utils import is_promptlayer_active

os.environ["OPENAI_API_KEY"] = config.openai_api_key


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
