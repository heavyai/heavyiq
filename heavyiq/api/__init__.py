import asyncio
import sys
from typing import Any

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from heavydb.exceptions import Error as HeavyDBError  # type: ignore
from langchain.globals import set_llm_cache
from starlette.exceptions import HTTPException

from heavyiq.api.handlers import exception_handler as exh
from heavyiq.api.middlewares import AsyncLoggingMiddleware
from heavyiq.api.models.error import ErrorResponse
from heavyiq.api.routes import bgrouter, defaultrouter, iqrouter, lcelrouter, llmrouter, streamrouter
from heavyiq.config import HeavyIQConfig, get_config
from heavyiq.langchain.exceptions import (
    GenerateTableMetadataException,
    NLtoAnswerException,
    NLtoSQLException,
    NLtoTableException,
)
from heavyiq.langchain.utils import InMemoryLLMCache, enable_telemetrics_for_free_edition, init_telemetrics
from heavyiq.logging_utils import get_heavyiq_logger, init_logs
from heavyiq.utils import SharedDictSingleton


def stripped_down_api() -> FastAPI:
    app = FastAPI(title="HeavyIQ")
    app.include_router(defaultrouter)
    return app


def app_initialize(config: HeavyIQConfig, config_path: str):
    """
    App initialization code which get excuted before gunicorn process fork upon using `--preload` option.
    """
    from heavyiq.langchain.heavydb import HeavyDB

    logger = get_heavyiq_logger()
    logger.info("Allocating Shared Dict....")
    # shared manager
    instance = SharedDictSingleton()
    instance.sput(SharedDictSingleton.Keys.ConfFilePath.name, config_path)
    logger.info(f"Shared Manager PID: {instance._manager._process.pid}")

    # always create a HeavyDB's multiprocessing.Manager instance (which was being used for shared cache) before gunicorn process fork
    # if we let it to happen on each worker process at the time of http request then
    # we might endup in request pending issue.
    HeavyDB.initialize()

    # initialize RAG DB
    from heavyrag.database import ragdb

    ragdb.create_tables()

    # LLM Cache
    if config.enable_llm_cache:
        set_llm_cache(InMemoryLLMCache())


def include_rag_routers(app: FastAPI) -> None:
    """
    Include RAG routers
    """
    from heavyiq.rag.router import doc_router, facts_db_router, table_router

    app.include_router(
        doc_router,
        prefix="/rag/documents",
        tags=["rag.documents"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    app.include_router(
        table_router,
        prefix="/rag/tables",
        tags=["rag.tables"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    app.include_router(
        facts_db_router,
        prefix="/rag/facts",
        tags=["rag.facts"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )


_config_provided = True


def create_app(config_path: str = "./config.toml") -> FastAPI:
    """
    create and return a FastAPI instance.

    :return FastAPI: instance of fastapi with custom openapi scehma.
    """
    global _config_provided
    try:
        with open(config_path, "r") as f:
            if "[iq]" not in f.read():
                print("No HeavyIQ configuration options detected; using defaults.")
                _config_provided = False
    except Exception:
        print("Provided config path is not valid.")
        _config_provided = False

    config = get_config(config_path, _config_provided)  # loads config using specified path

    # If IQ disabled in the config, don't go any further
    if config and config.disabled == True:
        # Figure out how to actually exit
        print("App disabled from config.toml, exiting.")
        sys.exit()

    init_logs()  # initializes logs using config
    init_telemetrics()  # initializes langsmith

    app = FastAPI(title="HeavyIQ")

    app_initialize(config, config_path)

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
    app.add_exception_handler(NLtoAnswerException, exh.nl_to_answer_exception_handler)
    app.add_exception_handler(GenerateTableMetadataException, exh.generate_table_metadata_exception_handler)
    app.add_exception_handler(NLtoTableException, exh.nl_to_tables_exception_handler)
    app.add_exception_handler(Exception, exh.unhandled_exception_handler)

    # Include your API routes
    app.include_router(defaultrouter)
    app.include_router(
        llmrouter,
        prefix="/llm",
        tags=["llm"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
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
    app.include_router(
        bgrouter,
        prefix="/bgtask",
        tags=["bgtask"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    # RAG routers
    include_rag_routers(app)

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

    @app.on_event("startup")
    async def initialize():
        """
        Code to be executed when application starts, ie on each worker process.
        """
        # initialize chains and RAG
        import heavyiq.lcel.chains
        import heavyrag.main
        from heavyiq.logging_utils import heavyiq_logger as logger

        global _config_provided

        async def enable_telemetrics_for_free_license_daemon():
            """
            This enables langsmith telemetrics for the free license by polling a shared multiprocessing dict.
            """

            shared_dict, max_retries, retry_count = SharedDictSingleton(), 20, 0
            while retry_count < max_retries:
                retry_count += 1
                await asyncio.sleep(2)
                license_edition = await shared_dict.get(SharedDictSingleton.Keys.HeavyDBLicenseEdition.name)
                if not license_edition:
                    continue

                logger.info(f"Found HeavyAI license edition, license_type: {license_edition}")

                if license_edition == "free":
                    logger.info("Enabling langsmith telemetrics for free edition.")
                    done = enable_telemetrics_for_free_edition()
                    if done:
                        logger.info("Successfully changed langsmith telemetrics and HeavyIQ configs for free edition.")
                    else:
                        logger.error("Failed to change langsmith telemetrics and HeavyIQ configs for free edition.")
                else:
                    if not _config_provided:
                        logger.error("Config must be provided when license edition is not free")
                        sys.exit()

                break
            else:
                logger.error(f"Failed to check HeavyAI license edition after {max_retries*2} seconds.")

        # run a background task to check license_edition got cached or not
        # if yes, and it's a free edition then enable langsmith telemetry
        asyncio.create_task(enable_telemetrics_for_free_license_daemon())

    @app.on_event("shutdown")
    async def shutdown():
        """
        Code to be executed before FastAPI application ends.
        """
        from heavyiq.logging_utils import heavyiq_logger as logger

        logger.info("Shutting down FastAPI worker.")

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
