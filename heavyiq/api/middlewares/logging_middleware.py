from typing import Any, Literal, Sequence

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from heavyiq.logging_utils import HeavyIQLogger, _AccessLogger, _get_access_logger, get_heavyiq_logger


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

    # paths where this middleware won't apply on
    disallowed_paths: Sequence[str] = ("/bgtask",)

    @staticmethod  # type: ignore
    def write_log_data(logs: list[Log]):
        """
        Supposed to write log data to terminal or to a file or both.

        Args:
            logs (list[Log]): list of logs
        """
        for log in logs:
            log_func = getattr(log.type, log.level)
            log_message = log.message
            if log.values:
                log_message = log_message % tuple(log.values)
            log_func(log_message, extra=log.extra)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path")
        if not path or any(i in path for i in self.disallowed_paths):
            # disbales this middleware for bgtask endpoints
            # otherwise we should endup in blocking future requests
            await self.app(scope, receive, send)
            return
        return await super().__call__(scope=scope, receive=receive, send=send)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Method responsible for dispatching request object to the appropriate handler.
        """
        is_api_request = "/api/v1/" in str(request.url)
        # Code executed before the request is processed

        logs: list[Log] = []
        heavyiq_logger: HeavyIQLogger = get_heavyiq_logger()
        app_logger: _AccessLogger = _get_access_logger()

        request_id = request.headers.get("x-request-id", "")

        with heavyiq_logger.logger.contextualize(request_id=request_id), app_logger.logger.contextualize(
            request_id=request_id
        ):
            if is_api_request:
                logs.append(
                    Log(
                        type=heavyiq_logger,
                        level="debug",
                        message="Request Path: %s",
                        values=[request.url.path],
                        extra={"request": request},
                    )
                )

            # write access log immediately when the request received
            self.write_log_data(
                [Log(type=app_logger, level="info", message="Request Received!", extra={"request": request})]
            )

            response = await call_next(request)
            # Code executed after the request has been processed
            # create access log
            logs.append(
                Log(
                    type=app_logger,
                    level="info",
                    message="Response Generated!",
                    extra={"response": response, "request": request},
                )
            )

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
