import logging
from typing import Any
from flask import request
from logging.handlers import RotatingFileHandler
import sys
from heavynl.config import get_log_config


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
        try:
            record.remote_addr = request.remote_addr or "-"
            record.username = request.remote_user or "-"
        except RuntimeError:
            # accessing through CLI
            record.remote_addr = "-"
            record.username = "CLI"
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
    This is the default logger which has to be used throughout the application.

    This log contain the following parts,

        - date time
        - remote_address
        - username
        - log level
        - file name path where the corresponding log method gets called along with the lineno
        - message

    Since it's the default logger, interactions made through the CLI are handled by this logger.

    By default all the logs produced by this logger gets stored inside "heavynl.log" file.
    """

    def __init__(self, name: str = "heavynl", log_file_path: str | None = None, level: str | None = None):
        super().__init__(
            name, level=level, log_file_path=log_file_path, filter=HeavyNLFilter(), formatter=get_default_formatter()
        )


class _AppLogger(BaseLogger):
    """
    Flask App logger class specifically used for logging http request calls which
    gets triggered after every api request.

    This logger is private and can't be used for general logging.
    Main purpose of this logger is to store all the information related to a http request/response cycle, including
        - remote_address
        - logged in username (prints "-" for unauthorized user)
        - date time
        - request_method
        - request_uri
        - protocol
        - referrer
        - user_agent
        - response status_code
        - response size in bytes
    Example:
        127.0.0.1 - - [2023-06-14 18:16:23] "POST /api/v1/query? HTTP/1.1" "http://localhost:5000/api/v1/ui/" "Safari/537.36" 500 21

    By default all the logs produced by this logger gets stored inside "access.log" file.
    """

    def __init__(self, name: str = "app", log_file_path: str | None = None, level: str | None = None):
        super().__init__(
            name, level=level, log_file_path=log_file_path, filter=AppFilter(), formatter=get_app_log_formatter()
        )


_app_logger = None
heavynl_logger = None
default_logger = None


# had to add this so the logs didn't init the config before we passed the config path
def init_logs():
    global _app_logger, heavynl_logger, default_logger
    LOG_CONFIG = get_log_config()

    _app_logger = _AppLogger(log_file_path=LOG_CONFIG.app_log_file, level=LOG_CONFIG.app_log_level)
    heavynl_logger = HeavyNLLogger(log_file_path=LOG_CONFIG.heavynl_log_file, level=LOG_CONFIG.heavynl_log_level)
    default_logger = heavynl_logger


def _get_app_logger() -> _AppLogger:
    global _app_logger
    if _app_logger is None:
        init_logs()
    return _app_logger  # type: ignore


def get_heavynl_logger() -> HeavyNLLogger:
    global heavynl_logger
    if heavynl_logger is None:
        init_logs()
    return heavynl_logger  # type: ignore
