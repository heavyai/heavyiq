from typing import Optional

from langchain.llms import OpenAI


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None
