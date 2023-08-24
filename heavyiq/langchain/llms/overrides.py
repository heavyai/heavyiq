from typing import Optional

from langchain.llms import OpenAI, VLLMOpenAI


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None


class OverrideVLLMOpenAI(VLLMOpenAI):
    context_window: Optional[int] = None
