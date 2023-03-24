import os
from logging import Logger
from typing import NamedTuple, Literal

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

ChatMessageRole = Literal["system", "assistant", "user"]
ChatMessage = NamedTuple("ChatMessage", [("role", ChatMessageRole), ("content", str)])

CHAT_MODEL = "gpt-3.5-turbo"


class ChatManager:
    """
    A class to manage messages used in communication with a ChatCompletion model.

    Attributes:
        model (str): The name of the chat model to use for generating responses.
        logger (Logger): The logger object for logging messages.
        _messages (list[ChatMessage]): A list of chat messages.
    """

    def __init__(self, logger: Logger, model: str = CHAT_MODEL):
        """
        Initializes a new instance of the ChatManager class.

        Args:
            logger (Logger): The logger object for logging messages.
            model (str, optional): The name of the chat model to use for generating responses. Defaults to CHAT_MODEL.
        """
        self.model = model
        self.logger = logger
        self._messages: list[ChatMessage] = []

    def add_message(self, role: ChatMessageRole, content: str):
        """
        Adds a new chat message with the given role and content.

        Args:
            role (ChatMessageRole): The role of the new message (system, assistant, or user).
            content (str): The content of the new message.
        """
        self._messages.append(ChatMessage(role, content))

    def reset_messages(self):
        """
        Resets the list of messages, removing all existing messages.
        """
        self._messages = []

    def formatted_conversation(self) -> str:
        """
        Returns the conversation as a formatted string with role and content.

        Returns:
            str: The formatted conversation string.
        """
        return "\n".join([f"[{m.role}] {m.content}" for m in self._messages])

    def prompt_ai(self) -> str:
        """
        Prompts the ChatCompletion model with the current messages and logs the input and output.
        The generated response is added to the list of messages as an 'assistant' message.

        Returns:
            str: The content of the generated response.
        """
        self.logger.info("==== Prompting OpenAI ====")
        self.logger.debug(self.formatted_conversation())
        response = openai.ChatCompletion.create(model=self.model, messages=self.messages)
        self.logger.info("==== OpenAI Response Received ====")
        content = response["choices"][0]["message"]["content"]
        self.logger.debug(f"Response: {content}")
        self.add_message("assistant", content)
        return content

    @property
    def messages(self) -> list[dict]:
        """
        Returns a list of messages as dictionaries, each containing role and content.

        Returns:
            list[dict]: A list of messages as dictionaries.
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]
