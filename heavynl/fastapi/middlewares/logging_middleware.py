import json
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from heavynl.logging_utils import _get_app_logger, get_heavynl_logger
from heavynl.fastapi.utils import AsyncIteratorWrapper


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Supports logging before and after each http request.
    """

    async def set_body(self, request: Request):
        """
        Avails the response body to be logged within a middleware as, it is generally not a standard practice.
        """
        receive_ = await request._receive()

        async def receive() -> Message:
            return receive_

        request._receive = receive

    async def _get_request_body(self, request: Request) -> str:
        """
        Gets the request body.
        """
        try:
            body = await request.json()
        except Exception:
            body = ""

        return body

    async def _get_response_body(self, response: Response) -> tuple[Response, str]:
        """
        Gets the response body for logging from Response object.
        """
        resp_body = [section async for section in response.__dict__["body_iterator"]]
        response.__setattr__("body_iterator", AsyncIteratorWrapper(resp_body))

        try:
            resp_body = json.loads(resp_body[0].decode())
        except Exception:
            resp_body = str(resp_body)

        return response, resp_body

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Method responsible for dispatching request object to the appropriate handler.
        """
        is_api_request = "/api/v1/" in str(request.url)
        # Code executed before the request is processed

        await self.set_body(request)
        if is_api_request:
            get_heavynl_logger().debug("Request Path: %s", request.url.path)
            get_heavynl_logger().debug("Request Body: %s", await request.json())

        response = await call_next(request)

        _get_app_logger().info("", extra={"response": response, "request": request})
        # Code executed after the request has been processed
        if not is_api_request:
            return response

        # log response body only for the api request
        try:
            reponse, response_body = await self._get_response_body(response)
            get_heavynl_logger().debug("Response Content: %s", response_body, extra={"request": request})
        except RuntimeError:
            # skip writing response data in-case of runtime error
            pass
        get_heavynl_logger().debug("Response Status Code: %d", response.status_code, extra={"request": request})
        return response
