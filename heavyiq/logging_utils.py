import copy
import getpass
import os
import socket
import time

from loguru import logger as loguru_logger

from heavyiq.config import get_config
from heavyiq.langchain.callbacks import AsyncLogFileCallbackHandler, FileCallbackHandler

from .loguru_logging import BaseAsyncLogger, access_formatter, iq_formatter

# remove existing handlers on the logger instance
loguru_logger.remove()


class HeavyIQLogger(BaseAsyncLogger):
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
        max_file_size: int = 104857600,
    ):
        iq_logger = copy.deepcopy(loguru_logger)
        super().__init__(
            name,
            iq_logger,
            level=level,
            format=iq_formatter,
            log_file_path=log_file_path,
            enable_console_logging=enable_console_logging,
            max_file_size=max_file_size,
        )
        self.info("HeavyIQ Logger initialized")

    def langchain_cb_handler(self, to_stdout: bool = True) -> FileCallbackHandler:
        """
        Returns a file callback handler created from the log file.

        Args:
            to_stdout (bool, optional): whether to log stdout as well. Defaults to True.

        Returns:
            FileCallbackHandler: callback handler mainly passed as callback for chains.
        """
        assert self.log_file_path
        return FileCallbackHandler(self.log_file_path, to_stdout=to_stdout)

    def async_langchain_cb_handler(self, to_stdout: bool = True) -> AsyncLogFileCallbackHandler:
        """
        Returns a file callback handler created from the log file.

        Args:
            to_stdout (bool, optional): whether to log stdout as well. Defaults to True.

        Returns:
            FileCallbackHandler: callback handler mainly passed as callback for chains.
        """
        assert self.log_file_path
        return AsyncLogFileCallbackHandler(self.log_file_path, to_stdout=to_stdout)


class _AccessLogger(BaseAsyncLogger):
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
        name: str = "access",
        log_file_path: str | None = None,
        level: str | None = None,
        enable_console_logging: bool = True,
        max_file_size: int = 104857600,
    ):
        access_logger = copy.deepcopy(loguru_logger)
        super().__init__(
            name,
            access_logger,
            level=level,
            format=access_formatter,
            log_file_path=log_file_path,
            max_file_size=max_file_size,
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
            max_file_size=LOG_CONFIG.access_log_max_file_size,
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
            max_file_size=LOG_CONFIG.heavyiq_log_max_file_size,
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
