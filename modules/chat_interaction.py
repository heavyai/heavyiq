"""
For handling chat-based interactions with the AI language model. This submodule should include classes and functions for
managing conversations, sending prompts, and processing responses from the AI model.
"""
from __future__ import annotations
import os
from logging import Logger
from typing import NamedTuple, Literal, TYPE_CHECKING

import openai

from modules.logging_utils import log_chat_conversation, default_logger
from modules.prompt_builder import build_chatgpt_system_answer_prompt, build_chatgpt_system_ask_prompt

if TYPE_CHECKING:
    from heavyai import Cursor

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

    def __init__(self, logger: Logger = default_logger, model: str = CHAT_MODEL):
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
        log_chat_conversation(self, self.logger)
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


def build_ask_manager(table_names: list[str], user_question: str) -> ChatManager:
    """
    Constructs a ChatManager object primed to prompt ChatGPT for an SQL query that can be used to answer the user's question.

    Args:
        table_names (list[str]): The names of the tables to include in the prompt.
        user_question (str): The user's question to be answered by the generated SQL query.

    Returns:
        ChatManager: A ChatManager object containing the system and user messages needed to prompt ChatGPT for an SQL query.
    """
    chat_manager = ChatManager()
    chat_manager.add_message("system", build_chatgpt_system_ask_prompt(table_names))
    chat_manager.add_message("user", user_question)
    return chat_manager


def build_answer_manager(sql_statement: str, user_question: str, sql_result: Cursor) -> ChatManager:
    """
    Constructs a ChatManager object primed to prompt ChatGPT for an answer to the user's question using the given SQL result.

    Args:
        sql_statement (str): The executed SQL statement.
        user_question (str): The user's question to be answered using the SQL result.
        sql_result (Cursor): A database cursor containing the result of the executed SQL query.

    Returns:
        ChatManager: A ChatManager object containing the system and user messages needed to prompt ChatGPT for an answer based on the SQL result.
    """
    chat_manager = ChatManager()
    chat_manager.add_message(
        "system",
        f"From the SQL query {sql_statement} in response to the following question: '{user_question}', we received the following result back from the database. Explain this result in plain English. Give a brief answer to the question without explanation of the SQL. Do not explain what the SQL means.",
    )
    chat_manager.add_message("user", build_chatgpt_system_answer_prompt(sql_result))
    return chat_manager
