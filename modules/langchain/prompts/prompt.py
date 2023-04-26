from typing import Any

from langchain.prompts import PromptTemplate
import promptlayer

from modules.langchain.utils import is_promptlayer_active


class LoggedPromptTemplate(PromptTemplate):
    name: str

    tags: list[str]

    version: int = 1
    """The version of the prompt. Increment went the prompt is changed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if is_promptlayer_active:
            try:
                # check to see if the prompt with specified version already exists
                promptlayer.prompts.get(self.name, version=self.version)
            except Exception:
                promptlayer.prompts.publish(self.name, self.tags, prompt_template=self)

    def track_request(self, request_id: Any, input_variables: dict):
        if is_promptlayer_active:
            promptlayer.track.prompt(request_id, self.name, input_variables, self.version)
