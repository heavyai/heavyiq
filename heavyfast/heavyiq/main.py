from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from heavydb.exceptions import Error as HeavyDBError

from heavyiq.api.routes import iqrouter
from heavyiq.core import exception_handler as exh


def create_app() -> FastAPI:
    """
    create and return a FastAPI instance.

    :return FastAPI: instance of fastapi with custom openapi scehma.
    """
    app = FastAPI(title="HeavyIQ")

    # add exception handlers
    app.add_exception_handler(RequestValidationError, exh.req_validation_handler)
    app.add_exception_handler(ValidationError, exh.validation_handler)
    app.add_exception_handler(AttributeError, exh.attribute_error_handler)

    app.add_exception_handler(HTTPException, exh.http_exception_handler)
    app.add_exception_handler(HeavyDBError, exh.heavydb_exception_handler)
    app.add_exception_handler(Exception, exh.unhandled_exception_handler)

    # Include your API routes
    app.include_router(iqrouter, prefix="/api/v1", tags=["api.v1"])

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="HeavyAI Endpoints",
            version="0.0.1",
            summary="Endpoints for doing NL operations.",
            description="",
            routes=app.routes,
        )
        openapi_schema["info"]["x-logo"] = {"url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"}
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    # custom openapi
    app.openapi = custom_openapi

    return app
