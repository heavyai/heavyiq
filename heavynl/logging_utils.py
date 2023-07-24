import logging
from logging.handlers import RotatingFileHandler
import getpass
import time
import socket
import os
from typing import Any
import sys
from fastapi import FastAPI


from heavynl.config import get_config


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


def is_fast_app(request: Any) -> bool:
    """
    Function to find whether fastapi app or not.
    """
    if request and getattr(request, "app", False) and isinstance(request.app, FastAPI):
        return True
    return False


# Define a custom filter to extract remote_addr and username from the Flask request object
class HeavyNLFilter(logging.Filter):
    """
    Filter which attaches extra attributes from request object to the record object.
    """

    def filter(self, record: Any) -> bool:
        try:
            request = getattr(record, "request")
            # handles both flask and fastapi logs
            if is_fast_app(request):
                record.remote_addr = request.client.host
                record.username = request.headers.get("X-Remote-User") or "-"
            else:
                record.remote_addr = request.remote_addr
                record.username = request.remote_user or "-"
        except RuntimeError:
            # accessing through CLI
            record.remote_addr = "-"
            record.username = "CLI"
        except AttributeError:
            # request object not been attached to the log record
            record.remote_addr = "-"
            record.username = "-"

        return True


class AppFilter(logging.Filter):
    """
    Filter out attributes from response object and attach it to the log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Get the request, response from record
        request: Any = getattr(record, "request", None)
        response: Any | None = getattr(record, "response", None)

        if is_fast_app(request):
            # fast api request
            record.remote_addr = request.client.host
            record.username = request.headers.get("X-Remote-User") or "-"
            record.request_method = request.method
            record.request_uri = str(request.url)
            record.referrer = request.headers.get("Referer")
            record.user_agent = request.headers.get("User-Agent")
            record.protocol = request.scope.get("scheme")
            record.status_code = response.status_code if response else "-"
            record.response_size = response.headers["Content-Length"] if response else "-"

        else:
            # flask request
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
        self.info("HeavyNL Logger initialized")


class _AccessLogger(BaseLogger):
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


_access_logger = None
heavynl_logger = None
default_logger = None


def get_log_name(lvl: str) -> str:
    """
    Constructs a log name based on the system's metadata and current time.

    Parameters:
    lvl (str): Log level.

    Returns:
    str: A log name in the following format:
    heavyiq.{hostname}.{username}.log.{log_level}.{timestamp}
    """
    h = socket.gethostname()
    u = getpass.getuser()
    t = time.strftime("%Y%m%d-%H%M%S")

    return f"heavyiq.{h}.{u}.log.{lvl}.{t}"


# had to add this so the logs didn't init the config before we passed the config path
def init_logs():
    global _access_logger, heavynl_logger, default_logger
    LOG_CONFIG = get_config()

    if _access_logger is None:
        access_log_name = get_log_name("ACCESS")
        data: str = LOG_CONFIG.data  # type: ignore
        log_dir = os.path.join(data, "log")

        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        _access_logger = _AccessLogger(
            log_file_path=os.path.join(log_dir, access_log_name), level=LOG_CONFIG.access_log_level
        )

        access_symlink = os.path.join(log_dir, "heavyiq.ACCESS")

        if os.path.islink(access_symlink):
            try:
                os.remove(access_symlink)
            except OSError as e:
                print(f"Failed to delete symlink: {e}")

        try:
            os.symlink(os.path.join("./", access_log_name), access_symlink)
        except OSError as e:
            print(f"Failed to create symlink: {e}")

    if heavynl_logger is None:
        app_log_name = get_log_name("APP")
        data: str = LOG_CONFIG.data  # type: ignore
        log_dir = os.path.join(data, "log")

        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        heavynl_logger = HeavyNLLogger(
            log_file_path=os.path.join(log_dir, app_log_name), level=LOG_CONFIG.heavyiq_log_level
        )

        app_symlink = os.path.join(log_dir, "heavyiq.ALL")

        if os.path.islink(app_symlink):
            try:
                os.remove(app_symlink)
            except OSError as e:
                print(f"Failed to delete symlink: {e}")

        try:
            os.symlink(os.path.join("./", app_log_name), app_symlink)
        except OSError as e:
            print(f"Failed to create symlink: {e}")

        default_logger = heavynl_logger


def _get_access_logger() -> _AccessLogger:
    global _access_logger
    if _access_logger is None:
        init_logs()
    return _access_logger  # type: ignore


def get_heavynl_logger() -> HeavyNLLogger:
    global heavynl_logger
    if heavynl_logger is None:
        init_logs()
    return heavynl_logger  # type: ignore
