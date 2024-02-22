from functools import wraps
from typing import Callable

from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from heavyiq.api.models import ErrorResponse
from heavyiq.langchain.exceptions import GenerateTableMetadataException, NLtoAnswerException, NLtoSQLException
from heavyiq.logging_utils import get_heavyiq_logger


def _build_error_response(title: str, msg: str) -> ErrorResponse:
    return ErrorResponse(error=f"{title}: {msg}")


def log_exception(func: Callable) -> Callable:
    """
    Used as a decorator for exception handler functions to log exceptions.
    """

    @wraps(func)
    async def wrapper(request: Request, exc: Exception) -> Response:
        logger = get_heavyiq_logger()
        import traceback

        print(traceback.format_exc())
        await run_in_threadpool(logger.exception, f"Exception Occured: {exc}")
        result = await func(request, exc)
        return result

    return wrapper


@log_exception
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("HTTP Exception", exc.detail)),
    )


@log_exception
async def heavydb_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("HeavyDB Error", str(exc))),
    )


@log_exception
async def value_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Value Error", str(exc))),
    )


@log_exception
async def type_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Type Error", str(exc))),
    )


@log_exception
async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Internal Server Error", "Internal Server Error")),
    )


@log_exception
async def attribute_error_handler(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("Attribute Error", str(exc))),
    )


@log_exception
async def nl_to_sql_exception_handler(request: Request, exc: NLtoSQLException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response("NLtoSQLException", exc.message or "")),
    )


@log_exception
async def nl_to_answer_exception_handler(request: Request, exc: NLtoAnswerException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response(exc.__class__.__name__, exc.message or "")),
    )


@log_exception
async def generate_table_metadata_exception_handler(request: Request, exc: GenerateTableMetadataException) -> Response:
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(_build_error_response(exc.__class__.__name__, exc.message or "")),
    )
