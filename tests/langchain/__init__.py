from typing import Optional, Any
import functools
import pytest
from unittest.mock import patch
from langchain.schema import BaseMessage, LLMResult, ChatGeneration, AIMessage, ChatResult, Generation, PromptValue
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.base import Callbacks
from langchain.chat_models import ChatOpenAI
from heavyiq.langchain.llms.overrides import OverrideOpenAI


class FakeChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI's fake.
    This class overrided _generate, _agenerate methods to return direct ChatResult isntead of calling openai server.
    Uses: Mainly used to test NLtoSQL chain.
    """

    generation_info: dict = {"finish_reason": "stop"}
    content: str = "Question: What is the total population in the USA according to the data in the usa_states table?\n\nSQLQuery: \nSELECT SUM(POPULATION) AS total_population\nFROM usa_states"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-openai"

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any | None = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.content), generation_info=self.generation_info)],
            llm_output={
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_name": self.model_name,
            },
        )

    def generate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self.content, generation_info=self.generation_info)]])


class FakeOpenAI(OverrideOpenAI):
    """
    Fake OpenAI llm.
    Mainly used to test NLtoAnswer chain.
    """

    generation_info = {"finish_reason": "stop"}
    content = "Total population in the USA according to the data in the usa_states table is 327514334"

    @property
    def _llm_type(self) -> str:
        return "fake-openai"

    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks | list[Callbacks] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self.content, generation_info=self.generation_info)]])

    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks | list[Callbacks] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self.content, generation_info=self.generation_info)]])
