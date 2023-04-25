import os

from langchain.llms.openai import OpenAI
from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI

from modules.config import config

os.environ["OPENAI_API_KEY"] = config.openai_api_key

use_promptlayer = False
if config.promptlayer_api_key is not None and config.promptlayer_api_key != "":
    print("THIS")
    os.environ["PROMPTLAYER_API_KEY"] = config.promptlayer_api_key
    use_promptlayer = True


def get_llm(tags: list[str] = [], **kwargs) -> OpenAI:
    if use_promptlayer:
        return PromptLayerOpenAI(pl_tags=tags, **kwargs)
    else:
        return OpenAI(**kwargs)


def get_chat_llm(tags: list[str] = [], **kwargs) -> ChatOpenAI:
    if use_promptlayer:
        return PromptLayerChatOpenAI(pl_tags=tags, **kwargs)
    else:
        return ChatOpenAI(**kwargs)
