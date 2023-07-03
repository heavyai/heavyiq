from __future__ import annotations
from pydantic import BaseModel, Extra
from typing import Any, Sequence, Union

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import BaseMessage
import promptlayer
from promptwatch import PromptWatch

from heavynl.config import get_config
from heavynl.langchain.utils import is_promptlayer_active


class LoggedPromptMixin:
    """
    Mixin class which helps to get or publish prompts.
    """

    def __post_init__(self):
        config = get_config()
        if config.promptwatch_api_key is not None and config.promptwatch_api_key != "":
            with PromptWatch(
                api_key=config.promptwatch_api_key, tracking_project=config.promptwatch_tracking_project
            ) as pw:
                pw.register_prompt_template(self.name, self, f"{self.version}.0")

        if is_promptlayer_active:
            try:
                # check to see if the prompt with specified version already exists
                promptlayer.prompts.get(self.name, version=self.version)
            except Exception:
                promptlayer.prompts.publish(self.name, self.tags, prompt_template=self.dict())

    def track_request(self, request_id: Any, input_variables: dict):
        if is_promptlayer_active:
            promptlayer.track.prompt(request_id, self.name, input_variables, self.version)


class BaseLoggedPrompt(LoggedPromptMixin):
    def __init__(self, name: str, tags: list[str], version: int = 1, **kwargs) -> None:
        super().__init__(name=name, tags=tags, version=version, **kwargs)
        self.__post_init__()


class LoggedPromptTemplate(BaseLoggedPrompt, PromptTemplate):
    """The version of the prompt. Increment went the prompt is changed."""

    name: str
    tags: list[str]
    version: int = 1


class LoggedChatPromptTemplate(BaseLoggedPrompt, ChatPromptTemplate):
    name: str
    tags: list[str]
    version: int = 1

    @property
    def _prompt_type(self) -> str:
        return "prompt"

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Union[BaseMessagePromptTemplate, BaseMessage]],
        name: str,
        tags: list[str],
        version: int = 1,
    ) -> LoggedChatPromptTemplate:
        input_vars = set()
        for message in messages:
            if isinstance(message, BaseMessagePromptTemplate):
                input_vars.update(message.input_variables)
        return cls(name, tags, version=version, input_variables=list(input_vars), messages=messages)
