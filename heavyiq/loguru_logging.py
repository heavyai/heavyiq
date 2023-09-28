import sys
from typing import Callable, Any
from copy import deepcopy
from starlette.requests import Request
from starlette.responses import Response
from loguru._defaults import LOGURU_FORMAT as DEFAULT_LOGURU_FORMAT
from loguru._logger import Logger
import datetime


class Rotator:
    def __init__(self, *, size):
        now = datetime.datetime.now()

        self._size_limit = size

    def should_rotate(self, message, file):
        file.seek(0, 2)
        if file.tell() + len(message) > self._size_limit:
            return True
        return False


def serialize(record: dict) -> dict:
    """
    Serializes a loguru record's expecially the request and response objects.
    """
    if record["extra"].get("has_serialized"):
        return record["extra"]

    record_extra = record["extra"].copy()
    # not yet serialized
    request, response = record_extra.get("request"), record_extra.get("response")
    request_dict, response_dict = {}, {"status_code": "-", "response_size": "-"}
    request_id = record_extra.get("request_id", None)
    if request and isinstance(request, Request):
        request_dict["remote_addr"] = request.client.host
        request_dict["username"] = request.headers.get("X-Remote-User") or "-"
        request_dict["request_method"] = request.method
        request_dict["request_uri"] = str(request.url)
        request_dict["referrer"] = request.headers.get("Referer")
        request_dict["user_agent"] = request.headers.get("User-Agent")
        request_dict["protocol"] = request.scope.get("scheme") or "-"
        request_id = request.headers.get("x-request-id")

    if response and isinstance(response, Response):
        response_dict["status_code"] = response.status_code or "-"
        response_dict["response_size"] = response.headers["Content-Length"] if response else "-"
        request_id = request.headers.get("x-request-id")

    request_id = request_id or "-"

    # remove request and response keys
    record_extra.pop("request", None)
    record_extra.pop("response", None)

    return {**record_extra, "request_id": request_id, **request_dict, **response_dict, "has_serialized": True}


def iq_formatter(record: dict) -> Callable[[Any], str]:
    """
    Formatter relevant to the HeavyIQ Logger.
    """
    # cleaned_extra = {k: v for k, v in extra.items() if k not in ["request", "response"]}
    record["extra"] = serialize(record)
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<yellow>{extra[logger_name]}</yellow> | "
        "<level>{level: <2}</level> | "
        "<level>{process: <2}</level> | "
        "<level>{thread: <2}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> (<blue>{extra[request_id]}</blue>) - <level>{message}</level>\n"
    )  # type: ignore


def access_formatter(record: dict) -> Callable[[Any], str]:
    """
    Formatter relevant to the Access Logger.
    """
    record["extra"] = serialize(record)

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<yellow>{extra[logger_name]}</yellow> | "
        "<level>{level: <2}</level> | "
        "<level>{process: <2}</level> | "
        "<level>{thread: <2}</level> | "
        "{extra[protocol]} {extra[request_method]} {extra[request_uri]} {extra[remote_addr]} {extra[username]} {extra[referrer]} {extra[user_agent]} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> (<blue>{extra[request_id]}</blue>) - <level>{message}</level> | "
        "{extra[response_size]} {extra[status_code]}\n"
    )  # type: ignore


class BaseAsyncLogger:
    """
    Base Loguru Logger.
    """

    def _get_filter(self, name: str) -> Callable[[Any], bool]:
        """
        Get filter function which used to filter records by logger name.
        """
        return lambda record: record["extra"]["logger_name"] == name

    def __init__(
        self,
        name: str,
        logger: Logger,
        level: str | None = "DEBUG",
        format: Callable | str = DEFAULT_LOGURU_FORMAT,  # type: ignore
        log_file_path: str | None = None,
        max_file_size: int | None = None,
        enable_console_logging: bool = True,
    ) -> None:
        self.name = name
        self.logger = logger.bind(logger_name=name)
        level = level or "DEBUG"
        self.log_file_path = log_file_path

        if enable_console_logging:
            self.logger.add(
                sys.stderr, level=level, format=format, enqueue=True, colorize=True, filter=self._get_filter(name)
            )
        if log_file_path:
            rotator = Rotator(size=max_file_size)
            self.logger.add(
                log_file_path,
                rotation=rotator.should_rotate,
                retention=5,
                level=level,
                format=format,
                enqueue=True,
                colorize=False,
                filter=self._get_filter(name),
            )

    def debug(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).debug(msg, **extra)

    def info(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).info(msg, **extra)

    def warning(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).warning(msg, **extra)

    def error(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).error(msg, **extra)

    def exception(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).exception(msg, **extra)

    def critical(self, msg, extra=None):
        extra = extra or {}
        self.logger.opt(depth=1).critical(msg, **extra)
