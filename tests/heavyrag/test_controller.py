# Test RAG controller module (unittests)
import os
from unittest.mock import PropertyMock, patch
from uuid import uuid4

import pytest

from heavyiq.config import HeavyIQConfig, get_config
from tests import aoverride_config


def get_uuid() -> str:
    return uuid4().hex


@pytest.fixture(scope="function")
def facts_data() -> list:
    return [
        (
            get_uuid(),
            "Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
        ),
        (
            get_uuid(),
            "Botanically, bananas qualify as berries because they develop from a flower with a single ovary and contain multiple seeds.",
        ),
        (
            get_uuid(),
            "Octopuses has two of the hearts pump blood to the gills, while the third pumps it to the rest of the body.",
        ),
    ]


@pytest.fixture(scope="function")
def chroma_controller_patch(heavyiq_config: HeavyIQConfig):
    """
    Overrided config w.r.t chroma vectorstore
    """
    with patch.object(heavyiq_config, "rag_vectordb_type", "chroma"), patch.object(
        heavyiq_config, "rag_chromadb_persist_dir", "/tmp/chroma"
    ):
        with patch("heavyrag.controller.CONFIG", heavyiq_config):
            from heavyrag.controller import get_controller

            controller = get_controller(heavyiq_config.rag_vectordb_type)
            assert controller.persist_dir == heavyiq_config.rag_chromadb_persist_dir
            return controller


@pytest.fixture(scope="function")
def faiss_controller_patch(heavyiq_config: HeavyIQConfig):
    """
    Overrided config w.r.t faiss vectorstore
    """
    with patch.object(heavyiq_config, "rag_vectordb_type", "faiss"), patch.object(
        heavyiq_config, "rag_faiss_persist_dir", "/tmp/faiss"
    ):
        with patch("heavyrag.controller.CONFIG", heavyiq_config):
            from heavyrag.controller import get_controller

            controller = get_controller(heavyiq_config.rag_vectordb_type)
            assert controller.persist_dir == heavyiq_config.rag_faiss_persist_dir
            yield controller


@pytest.mark.anyio
async def test_chroma_vectorstore_passes_facts_insert_and_list(chroma_controller_patch, facts_data):
    """
    Test insert/listing facts on chroma vectorstore.
    """
    first_dbname, second_dbname = "overture_wa", "overture_sa"
    *first_set, second_set = facts_data
    await chroma_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await chroma_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert all(i.get_content() in [k[1] for k in first_set] for i in first_nodes)
    assert all(i.get_content() in [second_set[1]] for i in second_nodes)


@pytest.mark.anyio
async def test_faiss_vectorstore_passes_facts_insert_and_list(faiss_controller_patch, facts_data):
    """
    Test insert/listing facts on faiss vectorstore.
    """
    first_dbname, second_dbname = "overture_wa", "overture_sa"
    *first_set, second_set = facts_data
    await faiss_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await faiss_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert all(i.get_content() in [k[1] for k in first_set] for i in first_nodes)
    assert all(i.get_content() in [second_set[1]] for i in second_nodes)
