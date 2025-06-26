from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import patch

import chromadb
import pytest

from heavyiq.config import HeavyIQConfig

if TYPE_CHECKING:
    from heavyrag.controller import BaseController


def delete_all_collections():
    from heavyrag.vector_stores.chroma import get_chroma_client

    client = get_chroma_client()

    # List all collections
    collections = client.list_collections()

    # Loop through all collections and delete them
    for collection in collections:
        collection_name = collection.name
        print(f"Deleting collection: {collection_name}")
        client.delete_collection(collection_name)

    print("All collections deleted.")


@pytest.fixture(scope="session")
def chroma_client(tmp_path_factory):
    # Define a persistent path for testing
    db_path = tmp_path_factory.mktemp("chroma_test_db")
    client = chromadb.PersistentClient(path=str(db_path))
    return client


@pytest.fixture(scope="function")
def chroma_controller_patch(heavyiq_config: HeavyIQConfig, chroma_client) -> Generator["BaseController", None, None]:
    """
    Overrided config w.r.t chroma vectorstore
    """
    chroma_temp_dir = "/tmp/chroma"

    with patch.object(heavyiq_config, "rag_vectordb_type", "chroma"), patch.object(
        heavyiq_config, "rag_chromadb_persist_dir", chroma_temp_dir
    ), patch("heavyrag.vector_stores.chroma.get_chroma_client", return_value=chroma_client):
        with patch("heavyrag.controller.CONFIG", heavyiq_config):
            from heavyrag.controller import get_controller

            controller = get_controller(heavyiq_config.rag_vectordb_type)
            assert controller.persist_dir == heavyiq_config.rag_chromadb_persist_dir
            yield controller
        # reset collections
        delete_all_collections()


@pytest.fixture(scope="function")
def faiss_controller_patch(heavyiq_config: HeavyIQConfig) -> Generator["BaseController", None, None]:
    """
    Overrided config w.r.t faiss vectorstore
    """
    faiss_temp_dir = "/tmp/faiss"
    with patch.object(heavyiq_config, "rag_vectordb_type", "faiss"), patch.object(
        heavyiq_config, "rag_faiss_persist_dir", faiss_temp_dir
    ):
        with patch("heavyrag.controller.CONFIG", heavyiq_config):
            from heavyrag.controller import get_controller

            controller = get_controller(heavyiq_config.rag_vectordb_type)
            assert controller.persist_dir == heavyiq_config.rag_faiss_persist_dir
            yield controller
    # resets all index data
    vs = controller.get_vectorstore()
    vs.reset()
