import logging
import typing
from abc import ABCMeta, abstractmethod
from logging import LogRecord
from logging.handlers import RotatingFileHandler
import getpass
import time
import socket
import os
from typing import Any, Self, Callable
import sys

from heavyiq.config import get_config


if typing.TYPE_CHECKING:
    from heavyiq.langchain.callbacks import AsyncLogFileCallbackHandler, FileCallbackHandler


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record: LogRecord, datefmt: str | None = None):
        """Adds milliseconds to the default formatter"""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
            return s + ".{:03d}".format(int((record.created % 1.0) * 1000))
        return super().formatTime(record, datefmt)


def get_default_formatter() -> logging.Formatter:
    log_format = "%(asctime)s %(remote_addr)s - %(username)s [%(levelname)s] %(pathname)s:%(lineno)d - %(message)s"
    return MillisecondFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")


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


# Define a custom filter to extract remote_addr and username from the request object
class HeavyIQFilter(logging.Filter):
    """
    Filter which attaches extra attributes from request object to the record object.
    """

    def filter(self, record: Any) -> bool:
        try:
            request = getattr(record, "request")
            record.remote_addr = request.client.host
            record.username = request.headers.get("X-Remote-User") or "-"
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

        record.remote_addr = request.client.host
        record.username = request.headers.get("X-Remote-User") or "-"
        record.request_method = request.method
        record.request_uri = str(request.url)
        record.referrer = request.headers.get("Referer")
        record.user_agent = request.headers.get("User-Agent")
        record.protocol = request.scope.get("scheme")
        record.status_code = response.status_code if response else "-"
        record.response_size = response.headers["Content-Length"] if response else "-"

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
        self.log_formatter = formatter
        self.log_file_path = ""
        if enable_console_logging:
            console_handler = get_console_handler(formatter=formatter, level=level)
            self.addHandler(console_handler)
        if log_file_path:
            file_handler = get_rotating_file_handler(log_file_path, formatter=formatter, level=level)
            self.addHandler(file_handler)
            self.log_file_path = log_file_path
        if filter:
            self.addFilter(filter)


class HeavyIQLogger(BaseLogger):
    """
    HeavyIQ Logger class specifically used for logging intermediate messages.
    This is the default logger which has to be used throughout the application.

    This log contain the following parts,

        - date time
        - remote_address
        - username
        - log level
        - file name path where the corresponding log method gets called along with the lineno
        - message

    Since it's the default logger, interactions made through the CLI are handled by this logger.

    By default all the logs produced by this logger gets stored inside "heavyiq.log" file.
    """

    def __init__(
        self,
        name: str = "heavyiq",
        log_file_path: str | None = None,
        level: str | None = None,
        enable_console_logging: bool = True,
    ):
        super().__init__(
            name,
            level=level,
            log_file_path=log_file_path,
            filter=HeavyIQFilter(),
            formatter=get_default_formatter(),
            enable_console_logging=enable_console_logging,
        )
        self.info("HeavyIQ Logger initialized")

    def langchain_cb_handler(self, to_stdout: bool = True) -> "FileCallbackHandler":
        """
        Returns a file callback handler created from the log file.

        Args:
            to_stdout (bool, optional): whether to log stdout as well. Defaults to True.

        Returns:
            FileCallbackHandler: callback handler mainly passed as callback for chains.
        """
        from heavyiq.langchain.callbacks import FileCallbackHandler

        return FileCallbackHandler(self.log_file_path, to_stdout=to_stdout)

    def async_langchain_cb_handler(self, to_stdout: bool = True) -> "AsyncLogFileCallbackHandler":
        """
        Returns a file callback handler created from the log file.

        Args:
            to_stdout (bool, optional): whether to log stdout as well. Defaults to True.

        Returns:
            AsyncLogFileCallbackHandler: callback handler mainly passed as callback for chains.
        """
        from heavyiq.langchain.callbacks import AsyncLogFileCallbackHandler

        return AsyncLogFileCallbackHandler(self.log_file_path, to_stdout=to_stdout)


class _AccessLogger(BaseLogger):
    """
    FastAPI App logger class specifically used for logging http request calls which
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

    def __init__(
        self,
        name: str = "app",
        log_file_path: str | None = None,
        level: str | None = None,
        enable_console_logging: bool = True,
    ):
        super().__init__(
            name,
            level=level,
            log_file_path=log_file_path,
            filter=AppFilter(),
            formatter=get_app_log_formatter(),
            enable_console_logging=enable_console_logging,
        )


_access_logger: _AccessLogger | None = None
heavyiq_logger: HeavyIQLogger | None = None
default_logger: HeavyIQLogger | None = None


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
    global _access_logger, heavyiq_logger, default_logger
    LOG_CONFIG = get_config()

    if _access_logger is None:
        access_log_name = get_log_name("ACCESS")
        data: str = LOG_CONFIG.data  # type: ignore
        log_dir = os.path.join(data, "log")

        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        _access_logger = _AccessLogger(
            log_file_path=os.path.join(log_dir, access_log_name),
            level=LOG_CONFIG.access_log_level,
            enable_console_logging=LOG_CONFIG.log_to_stdout,
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

    if heavyiq_logger is None:
        app_log_name = get_log_name("APP")
        data: str = LOG_CONFIG.data  # type: ignore
        log_dir = os.path.join(data, "log")

        # Ensure log_dir exists
        os.makedirs(log_dir, exist_ok=True)

        heavyiq_logger = HeavyIQLogger(
            log_file_path=os.path.join(log_dir, app_log_name),
            level=LOG_CONFIG.heavyiq_log_level,
            enable_console_logging=LOG_CONFIG.log_to_stdout,
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

        default_logger = heavyiq_logger


def _get_access_logger() -> _AccessLogger:
    global _access_logger
    if _access_logger is None:
        init_logs()
    return _access_logger  # type: ignore


def get_heavyiq_logger() -> HeavyIQLogger:
    global heavyiq_logger
    if heavyiq_logger is None:
        init_logs()
    return heavyiq_logger  # type: ignore


class LazyLogger(metaclass=ABCMeta):
    """
    Base logging utility that defers the initialization of logging resources until they are required.
    """

    error_msg: str = "Logging has not yet been initialized."

    @abstractmethod
    def instance(self) -> BaseLogger:
        ...

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """
        whenever an attribute is accessed, this method is supposed to call instance abstract method.
        """
        # should be a logger attribute
        return getattr(self.instance(), name)


class LazyIQLogger(LazyLogger):
    """
    Lazy Heavy IQ Logger.
    """

    def __init__(self) -> None:
        self._instance: HeavyIQLogger | None = None

    def instance(self: Self) -> HeavyIQLogger:
        if self._instance is None:
            # Perform the lazy loading here
            global heavyiq_logger

            if heavyiq_logger is None:
                raise AttributeError(self.error_msg)

            self._instance = heavyiq_logger

        return self._instance


class LazyAccessLogger(LazyLogger):
    """
    Lazy Access Logger.
    """

    def __init__(self) -> None:
        self._instance: _AccessLogger | None = None

    def instance(self: Self) -> _AccessLogger:
        if self._instance is None:
            # Perform the lazy loading here
            global _access_logger

            if _access_logger is None:
                raise AttributeError(self.error_msg)

            self._instance = _access_logger

        return self._instance


iq_logger = LazyIQLogger()  # using iq_logger on global scope should raise AttributeError
access_logger = LazyAccessLogger()
