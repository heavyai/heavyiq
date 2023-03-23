import os
from logging import Logger
from typing import NamedTuple, Literal

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

ChatMessageRole = Literal["system", "assistant", "user"]
ChatMessage = NamedTuple("ChatMessage", [("role", ChatMessageRole), ("content", str)])

CHAT_MODEL = "gpt-3.5-turbo"


class ChatManager:
    _messages: list[ChatMessage]

    def __init__(self, logger: Logger, model: str = CHAT_MODEL):
        self.model = model
        self.logger = logger
        self._messages = []

    def add_message(self, role: ChatMessageRole, content: str):
        self._messages.append(ChatMessage(role, content))

    def prompt_ai(self) -> str:
        self.logger.info("==== Prompting OpenAI ====")
        for message in self._messages:
            self.logger.debug(f"[{message.role}] {message.content}")
        response = openai.ChatCompletion.create(model=self.model, messages=self.messages)
        self.logger.info("==== OpenAI Response Received ====")
        content = response["choices"][0]["message"]["content"]
        self.logger.debug(f"Response: {content}")
        self.add_message("assistant", content)
        return content

    @property
    def messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]
