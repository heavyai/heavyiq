import os
import shutil
import tempfile
from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest
import sqlalchemy

from heavyiq.config import HeavyIQConfig

if TYPE_CHECKING:
    from heavyrag.controller import BaseController
    from heavyrag.database.base import Database


def _should_use_chroma(config: HeavyIQConfig) -> bool:
    """
    Determine if we should use ChromaDB instead of FAISS.
    
    Uses the rag_vectordb_type from config.
    """
    return config.rag_vectordb_type == "chroma"


def _check_chromadb_server(server_url: str) -> bool:
    """Check if ChromaDB server is running and accessible."""
    try:
        # ChromaDB has a /api/v2/heartbeat endpoint
        response = httpx.get(f"{server_url}/api/v2/heartbeat", timeout=5.0)
        return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


@pytest.fixture(scope="module")
def require_chromadb_server(heavyiq_config: HeavyIQConfig):
    """
    Module-scoped fixture that verifies ChromaDB server is running.
    
    Skips all tests that depend on this fixture if ChromaDB is not available.
    """
    if not _should_use_chroma(heavyiq_config):
        # Not using ChromaDB, skip this check
        yield
        return
    
    server_url = heavyiq_config.rag_chromadb_server_base
    if not _check_chromadb_server(server_url):
        pytest.skip(
            f"ChromaDB server not running at {server_url}. "
            f"Start it with: chroma run --host 127.0.0.1 --port 6271 --path /tmp/chromadb"
        )
    yield


@pytest.fixture(scope="module")
def ragdb(heavyiq_config: HeavyIQConfig, require_chromadb_server) -> Generator["Database", None, None]:
    """
    Initialize RAG database for tests.
    
    Uses the vectordb_type from config (config.toml).
    For tests on systems with FAISS TLS issues, set rag_vectordb_type = "chroma" in config.
    """
    # Use whatever the config file says - don't override
    # config.toml should have: rag_vectordb_type = "chroma" for systems with FAISS issues
    
    # Patch config in all modules that use it
    patches = [
        patch("heavyrag.database.base.config", heavyiq_config),
        patch("heavyrag.embed.CONFIG", heavyiq_config),
        patch("heavyrag.controller.CONFIG", heavyiq_config),
    ]
    
    try:
        for p in patches:
            p.start()

        from heavyiq.api import rag_initialize

        rag_initialize()
        from heavyrag.database import ragdb

        yield ragdb
    finally:
        for p in patches:
            p.stop()


@pytest.fixture(scope="module")
def faiss_rag_controller(heavyiq_config: HeavyIQConfig, ragdb: "Database") -> Generator["BaseController", None, None]:
    """
    Create a RAG controller for tests.
    
    Uses the vectordb backend specified in config (FAISS or ChromaDB).
    Patches all known import locations to ensure correct backend is used everywhere.
    """
    from heavyrag.controller import get_controller
    
    # Determine backend based on config (which was set by ragdb fixture)
    backend = heavyiq_config.rag_vectordb_type
    controller = get_controller(backend)
    
    # Patch rag_controller in ALL places where it's imported
    patches = [
        patch("heavyrag.controller.rag_controller", controller),
        patch("heavyiq.rag.router.rag_controller", controller),
        patch("heavyrag.main.rag_controller", controller),
    ]
    
    for p in patches:
        try:
            p.start()
        except AttributeError:
            pass  # Module not imported yet, skip
    
    try:
        yield controller
    finally:
        for p in patches:
            try:
                p.stop()
            except RuntimeError:
                pass  # Patch wasn't started


@pytest.fixture(scope="function")
async def pre_clean_table(heavyiq_config: HeavyIQConfig, ragdb: "Database", faiss_rag_controller: "BaseController"):
    """
    Fixture which helps to clean/delete all records in facts table and RAG index before running each testcase.
    """
    from heavyrag.database.models import FactsModel

    dbname = heavyiq_config.heavydb_dbname
    try:
        with ragdb.get_db() as session:
            FactsModel.delete_facts_by_database(db_session=session, heavydb_name=dbname)

        await faiss_rag_controller.delete_fact_nodes(dbname=dbname)
    except sqlalchemy.exc.OperationalError:
        pass
    yield
