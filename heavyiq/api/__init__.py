from typing import Any

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from heavydb.exceptions import Error as HeavyDBError  # type: ignore
from starlette.exceptions import HTTPException

from heavyiq.api.handlers import exception_handler as exh
from heavyiq.api.middlewares import AsyncLoggingMiddleware
from heavyiq.api.models.error import ErrorResponse
from heavyiq.api.routes import defaultrouter, iqrouter, lcelrouter, streamrouter
from heavyiq.config import get_config
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.utils import init_telemetrics
from heavyiq.logging_utils import init_logs


def stripped_down_api() -> FastAPI:
    app = FastAPI(title="HeavyIQ")
    app.include_router(defaultrouter)
    return app


def create_app(config_path: str = "./config.toml") -> FastAPI:
    """
    create and return a FastAPI instance.

    :return FastAPI: instance of fastapi with custom openapi scehma.
    """
    try:
        with open(config_path, "r") as f:
            if "[iq]" not in f.read():
                print("No HeavyIQ configuration options detected; service disabled.")
                return stripped_down_api()
    except Exception:
        print("Provided config path is not valid; service disabled.")
        return stripped_down_api()

    config = get_config(config_path)  # loads config using specified path
    init_logs()  # initializes logs using config
    init_telemetrics()  # initializes langsmith

    app = FastAPI(title="HeavyIQ")

    cors_origins = ["http://localhost"]

    # add middlewares
    app.add_middleware(AsyncLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    app.include_router(
        lcelrouter,
        prefix="/api/v1/lcel",
        tags=["api.v1.lcel"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    app.include_router(
        streamrouter,
        prefix="/api/v1/lcel/stream",
        tags=["api.v1.lcel.stream"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    if config.enable_debug_endpoints:
        from heavyiq.api.routes.debug_router import debug_router
        from heavyiq.api.routes.runnable_router import runnable_router

        app.include_router(
            debug_router,
            prefix="/debug",
            tags=["debug"],
            responses={
                500: {
                    "description": "Internal Server Error",
                    "model": ErrorResponse,
                }
            },
        )
        app.include_router(runnable_router, prefix="/runnable", tags=["runnable"])

    # check heavydb connection
    @app.on_event("startup")
    async def initialize():
        """
        Code to be executed when application starts.
        """
        import heavyiq.lcel.chains
        from heavyiq.logging_utils import heavyiq_logger as logger

        # TODO: Disabled for now, as no guarantee heavydb is running before heavyiq
        # logger.info("Connecting to heavydb...")
        # await run_in_threadpool(get_heavydb_license_claims, config)
        # logger.info("Successfully connected to heavydb...")

    @app.on_event("shutdown")
    async def shutdown():
        """
        Code to be executed before FastAPI application ends.
        """
        from heavyiq.logging_utils import heavyiq_logger as logger

        logger.info("Shutting down FastAPI app.")

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="HeavyIQ Endpoints",
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
    app.openapi = custom_openapi  # type: ignore

    return app
