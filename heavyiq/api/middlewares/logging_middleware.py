from pydantic import BaseModel
from typing import Literal, Any
from fastapi.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from heavyiq.logging_utils import _get_access_logger, get_heavyiq_logger, HeavyIQLogger, _AccessLogger
from starlette.types import ASGIApp, Scope, Receive, Send, Message


class Log(BaseModel):
    type: HeavyIQLogger | _AccessLogger
    level: Literal["debug", "info", "error"]
    message: str
    values: list = []
    extra: dict[str, Any] | None = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class AsyncLoggingMiddleware(BaseHTTPMiddleware):
    """
    Async logging middleware.
    Most of the time, logging involves writing the log content to a file (blocking I/O operation).
    So this middleware is supposed to log the content asynchorously without affecting the request/response cycle by
    running it as seperate Background Task.
    """

    @staticmethod
    def write_log_data(logs: list[Log]):
        """
        Supposed to write log data to terminal or to a file or both.

        Args:
            logs (list[Log]): list of logs
        """
        for log in logs:
            log_func = getattr(log.type, log.level)
            log_func(log.message, *log.values, extra=log.extra)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Method responsible for dispatching request object to the appropriate handler.
        """
        is_api_request = "/api/v1/" in str(request.url)
        # Code executed before the request is processed

        logs: list[Log] = []
        heavyiq_logger: HeavyIQLogger = get_heavyiq_logger()
        app_logger: _AccessLogger = _get_access_logger()

        if is_api_request:
            logs.append(Log(type=heavyiq_logger, level="debug", message="Request Path: %s", values=[request.url.path]))

        # write access log immediately when the request received
        await run_in_threadpool(
            self.write_log_data, [Log(type=app_logger, level="info", message="", extra={"request": request})]
        )

        response = await call_next(request)
        # Code executed after the request has been processed
        # create access log
        logs.append(Log(type=app_logger, level="info", message="", extra={"response": response, "request": request}))

        if is_api_request:
            logs.append(
                Log(
                    type=heavyiq_logger,
                    level="debug",
                    message="Response Status Code: %d",
                    values=[response.status_code],
                    extra={"request": request},
                )
            )

        response.background = BackgroundTask(self.write_log_data, logs)
        return response


class LogRequestsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.logger = get_heavyiq_logger()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_logs(message: Message):
            """Log every request info and response status code."""
            if message["type"] == "http.response.start":
                # request info is stored in the scope
                # status code is stored in the message
                await run_in_threadpool(
                    self.logger.info,
                    f'{scope["client"][0]}:{scope["client"][1]} - '
                    f'"{scope["method"]} {scope["path"]} '
                    f'{scope["scheme"]}/{scope["http_version"]}" '
                    f'{message["status"]}',
                )
            await send(message)

        await self.app(scope, receive, send_with_logs)
