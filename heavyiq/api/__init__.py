# Import faiss FIRST to avoid static TLS exhaustion error
# See: https://github.com/facebookresearch/faiss/issues/2595
import faiss  # noqa: F401

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from heavydb.exceptions import Error as HeavyDBError  # type: ignore
from langchain_core.globals import set_llm_cache
from starlette.exceptions import HTTPException

from heavyiq.api.handlers import exception_handler as exh
from heavyiq.api.handlers import handle_submit_feedback
from heavyiq.api.middlewares import AsyncLoggingMiddleware
from heavyiq.api.models.error import ErrorResponse
from heavyiq.api.routes import bgrouter, defaultrouter, lcelrouter, llmrouter, streamrouter
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
from heavyiq import shared_state


def stripped_down_api() -> FastAPI:
    app = FastAPI(title="HeavyIQ")
    app.include_router(defaultrouter)
    return app


def rag_create_tables():
    """
    Create RAG database tables. Safe to call before fork (SQLite create_all is idempotent).
    """
    from heavyrag.database import ragdb
    ragdb.create_tables()


def rag_initialize():
    """
    Initialize heavyrag models and clients.
    """
    from heavyrag import initialize_rag
    # initializes embedding model and vector store client
    initialize_rag()


def app_initialize(config: HeavyIQConfig, config_path: str):
    """
    App initialization code which gets executed before gunicorn process fork upon using `--preload` option.
    
    With --preload, this runs BEFORE Gunicorn's on_starting hook, so we start the
    shared state manager here to ensure it's available before storing any data.
    """
    import os
    from heavyiq.langchain.heavydb import HeavyDB
    from heavyiq.langchain.utils import initialize_tokenizer

    logger = get_heavyiq_logger()
    
    # Start shared state manager FIRST, before storing any data.
    # This must happen before any shared_state.put() calls.
    # In standalone mode (uvicorn), this will be a no-op and auto-init will use local dict.
    try:
        shared_state.start_manager()
        logger.info(f"Started shared state manager at {shared_state.get_manager_info()['manager_address']}")
    except Exception as e:
        logger.warning(f"Could not start shared state manager: {e}. Using standalone mode.")
        shared_state.init_standalone()
    
    # Store config path in shared state
    logger.info("Storing config path in shared state...")
    shared_state.put(shared_state.SharedStateKeys.ConfFilePath.name, config_path)
    
    # Also store in environment as backup
    os.environ["HEAVYIQ_CONFIG_PATH"] = config_path
    
    manager_info = shared_state.get_manager_info()
    if manager_info["is_standalone"]:
        logger.info("Running in standalone mode (no shared manager)")
    else:
        logger.info(f"Using shared manager at {manager_info['manager_address']}")

    # Initialize all caches before Gunicorn fork.
    # These use shared_state internally for cross-process sharing.
    from heavyiq.utils import TablesCache
    TablesCache.initialize()
    HeavyDB.initialize()
    initialize_tokenizer()

    # LLM Cache
    if config.enable_llm_cache:
        set_llm_cache(InMemoryLLMCache())
    if config.enable_rag:
        # Always create RAG database tables before fork (idempotent, safe for SQLite)
        logger.info("Creating RAG database tables...")
        rag_create_tables()
        
        # Both FAISS and ChromaDB are initialized after fork for consistency
        # - Gunicorn: post_fork hook initializes RAG
        # - Uvicorn standalone: FastAPI startup event initializes RAG
        # Note: `import faiss` is still at module level to get TLS slots first
        logger.info(f"RAG with {config.rag_vectordb_type} will be initialized after fork/startup")


def add_rag_db_exception_handlers(app: FastAPI) -> None:
    """
    Adding RAG DB exception handlers to the fastapi app.
    """
    from heavyrag.database import RAGDBIntegrityError

    app.add_exception_handler(RAGDBIntegrityError, exh.ragdb_integrity_exception_handler)


def include_rag_routers(app: FastAPI) -> None:
    """
    Include RAG routers
    """
    from heavyiq.rag.router import collection_router, doc_router, facts_db_router, table_router

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
        prefix="/rag/snippets",
        tags=["rag.snippets"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )
    app.include_router(
        collection_router,
        prefix="/rag/collection",
        tags=["rag.collection"],
        responses={
            500: {
                "description": "Internal Server Error",
                "model": ErrorResponse,
            }
        },
    )


_config_provided = True


def create_lifespan(config: HeavyIQConfig) -> Any:
    """
    Create a lifespan context manager for FastAPI.
    
    This replaces the deprecated @app.on_event("startup") and @app.on_event("shutdown") decorators.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """
        Lifespan context manager for FastAPI application.
        
        Startup: Initializes RAG, chains, and background tasks.
        Shutdown: Cleans up resources.
        """
        # === STARTUP ===
        import heavyiq.lcel.chains
        from heavyiq.logging_utils import heavyiq_logger as logger
        
        # Initialize RAG (FAISS or ChromaDB) if not already done
        # - Gunicorn: post_fork may have already initialized, so we check _initialized flag
        # - Uvicorn standalone: post_fork is not called, so we initialize here
        if config.enable_rag:
            try:
                from heavyrag.controller import rag_controller
                from heavyiq.api import rag_initialize
                
                if not hasattr(rag_controller, '_initialized') or not rag_controller._initialized:
                    logger.info(f"Initializing RAG with {config.rag_vectordb_type} (FastAPI lifespan startup)")
                    rag_initialize()
                    rag_controller._initialized = True
                else:
                    logger.info("RAG already initialized (skipping in FastAPI lifespan)")
            except Exception as e:
                logger.warning(f"RAG initialization in lifespan startup: {e}")
        
        async def enable_telemetrics_for_free_license_daemon():
            """
            This enables langsmith telemetrics for the free license by polling a shared multiprocessing dict.
            """
            try:
                max_retries, retry_count = 20, 0
                while retry_count < max_retries:
                    retry_count += 1
                    await asyncio.sleep(2)
                    try:
                        license_edition = shared_state.get(shared_state.SharedStateKeys.HeavyDBLicenseEdition.name)
                    except (EOFError, OSError, ConnectionError) as e:
                        logger.debug(f"Shared dict unavailable: {e}")
                        return
                    
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
                    
                    break
                else:
                    logger.error(f"Failed to check HeavyAI license edition after {max_retries*2} seconds.")
            except Exception as e:
                logger.warning(f"Telemetry daemon error (non-fatal): {e}")
        
        # Start background task for license edition check
        background_task = asyncio.create_task(enable_telemetrics_for_free_license_daemon())
        
        logger.info("FastAPI application started")
        
        yield  # Application runs here
        
        # === SHUTDOWN ===
        logger.info("Shutting down FastAPI worker")
        
        # Cancel background task to prevent cleanup errors
        if not background_task.done():
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
    
    return lifespan


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

    app_initialize(config, config_path)
    
    # Create lifespan context manager for startup/shutdown
    lifespan = create_lifespan(config)
    app = FastAPI(title="HeavyIQ", lifespan=lifespan)

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
    app.add_api_route(
        path="/api/v1/submit-feedback",
        endpoint=handle_submit_feedback,
        methods=["POST"],
        include_in_schema=True,
    )
    # @deprecated
    # now we have endpoints for syncing table and facts/snippets index
    # app.include_router(
    #     bgrouter,
    #     prefix="/bgtask",
    #     tags=["bgtask"],
    #     responses={
    #         500: {
    #             "description": "Internal Server Error",
    #             "model": ErrorResponse,
    #         }
    #     },
    # )
    if config.enable_rag:
        add_rag_db_exception_handlers(app)
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
