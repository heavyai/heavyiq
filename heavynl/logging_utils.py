import logging
import sys


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
