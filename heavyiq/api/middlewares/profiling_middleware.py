from fastapi.responses import JSONResponse
from pyinstrument import Profiler
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class ProfilingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for doing profiling on endpoints.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        profiling = request.query_params.get("profile", False)
        if profiling:
            profiler = Profiler(interval=0.1, async_mode="enabled")
            profiler.start()
            await call_next(request)
            profiler.stop()
            return JSONResponse({"path": profiler.open_in_browser(timeline=False)})
        else:
            return await call_next(request)
