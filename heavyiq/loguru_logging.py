import logging
import sys
from typing import Callable, Any
from copy import deepcopy
from loguru import logger as loguru_logger
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


def formatter(record):
    # Note this function returns the string to be formatted, not the actual message to be logged
    extra = record["extra"].copy()
    request, response = extra.get("request"), extra.get("response")
    cleaned_extra = {k: v for k, v in extra.items() if k not in ["request", "response"]}
    if not (request or response):
        record["extra"] = cleaned_extra
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<yellow>{extra[logger_name]}</yellow> | "
            "<level>{level: <2}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
        )
    request_method, remote_addr, request_uri, username, referrer, user_agent, protocol, status_code, response_size = (
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    )
    if request:
        remote_addr = request.client.host
        username = request.headers.get("X-Remote-User") or "-"
        request_method = request.method
        request_uri = str(request.url)
        referrer = request.headers.get("Referer")
        user_agent = request.headers.get("User-Agent")
        protocol = request.scope.get("scheme") or "-"
    if response:
        status_code = response.status_code or "-"
        response_size = response.headers["Content-Length"] if response else "-"

    extra_dict = {
        "protocol": protocol,
        "request_method": request_method,
        "request_uri": request_uri,
        "remote_addr": remote_addr,
        "username": username,
        "referrer": referrer,
        "user_agent": user_agent,
        "response_size": response_size,
        "status_code": status_code,
    }

    record["extra"] = {**cleaned_extra, **extra_dict}

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<yellow>{extra[logger_name]}</yellow> | "
        "<level>{level: <2}</level> | "
        "{extra[protocol]} {extra[request_method]} {extra[request_uri]} {extra[remote_addr]} {extra[username]} {extra[referrer]} {extra[user_agent]} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | "
        "{extra[response_size]} {extra[status_code]}\n"
    )


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
                sys.stderr, level=level, format=formatter, enqueue=True, colorize=True, filter=self._get_filter(name)
            )
        if log_file_path:
            rotator = Rotator(size=max_file_size)
            self.logger.add(
                log_file_path,
                rotation=rotator.should_rotate,
                level=level,
                format=formatter,
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
