import logging
from typing import Any
from flask import request
from logging.handlers import RotatingFileHandler
import sys
from heavynl.config import get_log_config

LOG_CONFIG = get_log_config()


def get_default_formatter() -> logging.Formatter:
    log_format = "%(asctime)s %(remote_addr)s - %(username)s [%(levelname)s] %(pathname)s:%(lineno)d - %(message)s"
    return logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")


def get_app_log_formatter() -> logging.Formatter:
    log_format = '%(remote_addr)s - %(username)s [%(asctime)s] "%(request_method)s %(request_uri)s %(protocol)s" "%(referrer)s" "%(user_agent)s" %(status_code)s %(response_size)s'
    return logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")


def get_rotating_file_handler(
    log_file_path: str, formatter: logging.Formatter | None = None, level: str = "DEBUG"
) -> RotatingFileHandler:
    formatter = formatter or get_default_formatter()
    file_handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    return file_handler


def get_console_handler(formatter: logging.Formatter | None = None, level: str = "DEBUG") -> logging.StreamHandler:
    formatter = formatter or get_default_formatter()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level)
    console_handler.setFormatter(formatter)

    return console_handler


# Define a custom filter to extract remote_addr and username from the Flask request object
class HeavyNLFilter(logging.Filter):
    """
    Filter which attaches extra attributes from request object to the record object.
    """

    def filter(self, record: Any) -> bool:
        record.remote_addr = request.remote_addr
        record.username = request.remote_user or "-"
        return True


class AppFilter(logging.Filter):
    """
    Filter out attributes from response object and attach it to the log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Get the response from the Flask context
        response: Any | None = getattr(request, "response", None)
        record.remote_addr = request.remote_addr
        record.username = request.remote_user or "-"
        record.request_method = request.method
        record.request_uri = request.full_path
        record.referrer = request.referrer
        record.user_agent = request.user_agent
        record.protocol = request.environ.get("SERVER_PROTOCOL", "-")
        # Include the response in the log record
        record.status_code = response.status_code if response else "-"
        record.response_size = response.content_length if response else "-"

        return True


def create_custom_logger(name: str, log_file_path: str | None = None) -> logging.Logger:
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
        # Create a formatter for the log messages
        formatter = get_default_formatter()
        console_handler = get_console_handler(formatter=formatter)
        logger.addHandler(console_handler)

        # Optional: Create a rotating file handler if a log file path is provided
        if log_file_path:
            file_handler = get_rotating_file_handler(log_file_path, formatter=formatter)
            logger.addHandler(file_handler)

    return logger


default_logger = create_custom_logger("default")


class BaseLogger(logging.Logger):
    def __init__(
        self,
        name: str,
        level: str | None = "DEBUG",
        log_file_path: str | None = None,
        filter: logging.Filter | None = None,
        formatter: logging.Formatter | None = None,
        enable_console_logging: bool = True,
    ) -> None:
        level = level or "DEBUG"
        super().__init__(name, level)
        formatter = formatter or get_default_formatter()
        if enable_console_logging:
            console_handler = get_console_handler(formatter=formatter, level=level)
            self.addHandler(console_handler)
        if log_file_path:
            file_handler = get_rotating_file_handler(log_file_path, formatter=formatter, level=level)
            self.addHandler(file_handler)
        if filter:
            self.addFilter(filter)


class HeavyNLLogger(BaseLogger):
    """
    HeavyNL Logger class specifically used for logging intermediate messages.
    """

    def __init__(self, name: str = "heavynl", log_file_path: str | None = None, level: str | None = None):
        super().__init__(
            name, level=level, log_file_path=log_file_path, filter=HeavyNLFilter(), formatter=get_default_formatter()
        )


class AppLogger(BaseLogger):
    """
    Flask App logger class specifically used for logging http request calls.
    """

    def __init__(self, name: str = "app", log_file_path: str | None = None, level: str | None = None):
        super().__init__(
            name, level=level, log_file_path=log_file_path, filter=AppFilter(), formatter=get_app_log_formatter()
        )


app_logger = AppLogger(log_file_path=LOG_CONFIG.app_log_file, level=LOG_CONFIG.app_log_level)
heavynl_logger = HeavyNLLogger(log_file_path=LOG_CONFIG.heavynl_log_file, level=LOG_CONFIG.heavynl_log_level)
