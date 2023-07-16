import logging

from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from heavynl.fastapi.models import ErrorResponse
from heavynl.langchain.exceptions import NLtoSQLException


logger = logging.getLogger(__name__)


def _build_error_response(title: str, msg: str) -> ErrorResponse:
    return ErrorResponse(error=f"{title}: {msg}")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("HTTP Exception", exc.detail)),
    )


async def heavydb_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("HeavyDB Error", str(exc))),
    )


async def value_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Value Error", str(exc))),
    )


async def type_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Type Error", str(exc))),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Internal Server Error", "Internal Server Error")),
    )


async def attribute_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Attribute Error", str(exc))),
    )


async def nl_to_sql_exception_handler(request: Request, exc: NLtoSQLException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("NLtoSQLException", exc.message or "")),
    )
