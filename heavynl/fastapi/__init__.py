from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.utils import get_openapi
from heavydb.exceptions import Error as HeavyDBError
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from heavynl.config import get_config, get_heavydb_license_claims
from heavynl.fastapi.handlers import exception_handler as exh
from heavynl.fastapi.middlewares import LoggingMiddleware, AsyncLoggingMiddleware
from heavynl.fastapi.routes import defaultrouter, iqrouter
from heavynl.fastapi.models.error import ErrorResponse
from heavynl.langchain.exceptions import NLtoSQLException
from heavynl.logging_utils import init_logs


def create_app(config_path: str = "./config.toml") -> FastAPI:
    """
    create and return a FastAPI instance.

    :return FastAPI: instance of fastapi with custom openapi scehma.
    """
    config = get_config(config_path)  # loads config using specified path
    init_logs()  # initializes logs using config

    app = FastAPI(title="HeavyIQ")

    cors_origins = [
        "http://localhost",
        "http://localhost:8000",
    ]

    # add middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AsyncLoggingMiddleware)

    # add exception handlers
    app.add_exception_handler(AttributeError, exh.attribute_error_handler)
    app.add_exception_handler(HTTPException, exh.http_exception_handler)
    app.add_exception_handler(HeavyDBError, exh.heavydb_exception_handler)
    app.add_exception_handler(ValueError, exh.value_error_handler)
    app.add_exception_handler(TypeError, exh.type_error_handler)
    app.add_exception_handler(NLtoSQLException, exh.nl_to_sql_exception_handler)
    app.add_exception_handler(Exception, exh.unhandled_exception_handler)

    # Include your API routes
    app.include_router(defaultrouter)
    app.include_router(
        iqrouter,
        prefix="/api/v1",
        tags=["api.v1"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )

    # check heavydb connection
    @app.on_event("startup")
    async def initialize():
        """
        Code to be executed when application starts.
        """
        from heavynl.logging_utils import heavynl_logger as logger

        logger.info("Connecting to heavydb...")
        await run_in_threadpool(get_heavydb_license_claims, config)
        logger.info("Successfully connected to heavydb...")

    @app.on_event("shutdown")
    async def shutdown():
        """
        Code to be executed before FastAPI application ends.
        """
        from heavynl.logging_utils import heavynl_logger as logger

        logger.info("Shutting down FastAPI app.")

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="HeavyAI Endpoints",
            version="0.0.1",
            summary="Endpoints for doing NL operations.",
            description="",
            routes=app.routes,
        )

        # hard way to remove certain schemas from default openapi schema
        def remove_key(dictionary: dict, keys_to_remove: list):
            for key, value in list(dictionary.items()):
                if key in keys_to_remove:
                    dictionary.pop(key)
                elif isinstance(value, dict):
                    remove_key(value, keys_to_remove)

        remove_key(openapi_schema, ["422", "HTTPValidationError", "ValidationError"])

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    # custom openapi
    app.openapi = custom_openapi

    return app
