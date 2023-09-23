import json
from pydantic import BaseModel
from typing import Literal, Any
from fastapi.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from heavyiq.logging_utils import _get_access_logger, get_heavyiq_logger, HeavyIQLogger, _AccessLogger
from heavyiq.api.utils import AsyncIteratorWrapper


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Supports logging before and after each http request.
    """

    async def _get_response_body(self, response: Response) -> tuple[Response, str]:
        """
        Gets the response body for logging from Response object.
        """
        resp_body_sections = [section async for section in response.__dict__["body_iterator"]]
        response.__setattr__("body_iterator", AsyncIteratorWrapper(resp_body_sections))

        try:
            resp_body: str = json.loads(resp_body_sections[0].decode())  # type: ignore
        except Exception:
            resp_body = str(resp_body_sections)

        return response, resp_body

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Method responsible for dispatching request object to the appropriate handler.
        """
        is_api_request = "/api/v1/" in str(request.url)
        # Code executed before the request is processed

        if is_api_request:
            get_heavyiq_logger().debug("Request Path: %s", request.url.path)

        response = await call_next(request)

        _get_access_logger().info("", extra={"response": response, "request": request})
        # Code executed after the request has been processed
        if not is_api_request:
            return response

        # log response body only for the api request
        try:
            reponse, response_body = await self._get_response_body(response)
            get_heavyiq_logger().debug("Response Content: %s", response_body, extra={"request": request})
        except RuntimeError:
            # skip writing response data in-case of runtime error
            pass
        get_heavyiq_logger().debug("Response Status Code: %d", response.status_code, extra={"request": request})
        return response


class Log(BaseModel):
    type: HeavyIQLogger | _AccessLogger
    level: Literal["debug", "info", "error"]
    message: str
    values: list = []
    extra: dict[str, Any] | None = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class AsyncLoggingMiddleware(LoggingMiddleware):
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

        # log response body only for the api request
        if is_api_request:
            try:
                reponse, response_body = await self._get_response_body(response)
                logs.append(
                    Log(
                        type=heavyiq_logger,
                        level="debug",
                        message="Response Content: %s",
                        values=[response_body],
                        extra={"request": request},
                    )
                )
            except RuntimeError:
                # skip writing response data in-case of runtime error
                pass
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
