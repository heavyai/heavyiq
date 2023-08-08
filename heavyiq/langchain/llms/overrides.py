from typing import Optional

from langchain.llms.promptlayer_openai import PromptLayerOpenAI
from langchain.llms import OpenAI


class OverridePromptLayerOpenAI(PromptLayerOpenAI):
    context_window: Optional[int] = None


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None
