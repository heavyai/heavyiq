from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from modules.chat_interaction import ChatManager


def create_custom_logger(name: str) -> logging.Logger:
    """
    Creates a custom logger with predefined formatters and handlers.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The custom logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


default_logger = create_custom_logger("heavynl")


def log_chat_conversation(chat_manager: ChatManager, logger: logging.Logger = default_logger):
    """
    Logs an entire chat conversation.

    Args:
        chat_manager (ChatManager): The ChatManager instance containing the chat conversation.
        logger (logging.Logger): The logger to use for logging the conversation.
    """
    logger.info("==== Chat Conversation ====")
    for message in chat_manager._messages:
        logger.info(f"[{message.role}] {message.content}")
    logger.info("============================")
