import logging

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR

logger = logging.getLogger(__name__)


def _build_validation_errors(exc: RequestValidationError, title: str) -> dict:
    return {
        "errors": [
            {
                "title": title,
                "source": "/".join(map(str, error["loc"])),
                "msg": error["msg"],
            }
            for error in exc.errors()
        ]
    }


def _build_error_dict(title: str, msg: str) -> dict:
    return {"error": f"{title}: {msg}"}


async def req_validation_handler(request: Request, exc: RequestValidationError) -> Response:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(_build_validation_errors(exc, "Request Validation Error")),
    )


async def validation_handler(request: Request, exc: RequestValidationError) -> Response:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(_build_validation_errors(exc, "Validation Error")),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(_build_error_dict("HTTP Exception", exc.detail)),
    )


async def heavydb_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_dict("HeavyDB Error", str(exc))),
    )


async def value_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_dict("Value Error", str(exc))),
    )


async def type_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_dict("Type Error", str(exc))),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_dict("Internal Server Error", "Internal Server Error")),
    )


async def attribute_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(_build_error_dict("Attribute Error", str(exc))),
    )
