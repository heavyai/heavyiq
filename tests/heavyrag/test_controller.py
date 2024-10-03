# Test RAG controller module (unittests)
import os
import shutil
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


@pytest.fixture(scope="function")
def chroma_controller_patch(heavyiq_config: HeavyIQConfig):
    """
    Overrided config w.r.t chroma vectorstore
    """
    chroma_temp_dir = "/tmp/chroma"
    with patch.object(heavyiq_config, "rag_vectordb_type", "chroma"), patch.object(
        heavyiq_config, "rag_chromadb_persist_dir", chroma_temp_dir
    ):
        with patch("heavyrag.controller.CONFIG", heavyiq_config):
            from heavyrag.controller import get_controller

            controller = get_controller(heavyiq_config.rag_vectordb_type)
            assert controller.persist_dir == heavyiq_config.rag_chromadb_persist_dir
            yield controller
    # reset collections
    delete_all_collections()


@pytest.fixture(scope="function")
def faiss_controller_patch(heavyiq_config: HeavyIQConfig):
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
async def test_chroma_vectorstore_passes_facts_delete(chroma_controller_patch, facts_data):
    """
    Test insert/listing facts on chroma vectorstore.
    """
    first_dbname, second_dbname = "overture_wa", "overture_sa"
    *first_set, second_set = facts_data

    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert not (first_nodes)
    assert not (second_nodes)
    # insert nodes
    await chroma_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await chroma_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    # check nodes after insertion
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert first_nodes
    assert second_nodes
    # single node deletion
    first_fact_id = first_nodes[0].node_id
    await chroma_controller_patch.delete_fact_nodes(dbname=first_dbname, fact_ids=[first_fact_id])
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    assert len(first_nodes) == 1
    # delete all nodes
    await chroma_controller_patch.delete_fact_nodes(dbname=second_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert not (second_nodes)


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


@pytest.mark.anyio
async def test_faiss_vectorstore_passes_facts_delete(faiss_controller_patch, facts_data):
    """
    Test insert/listing facts on chroma vectorstore.
    """
    first_dbname, second_dbname = "overture_wa", "overture_sa"
    *first_set, second_set = facts_data

    vs = faiss_controller_patch.get_vectorstore()
    vs.reset()

    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert not (first_nodes)
    assert not (second_nodes)
    # insert nodes
    await faiss_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await faiss_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    # check nodes after insertion
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert first_nodes
    assert second_nodes
    # single node deletion
    first_fact_id = first_nodes[0].node_id
    await faiss_controller_patch.delete_fact_nodes(dbname=first_dbname, fact_ids=[first_fact_id])
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    assert len(first_nodes) == 1
    # delete all nodes
    await faiss_controller_patch.delete_fact_nodes(dbname=second_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert not (second_nodes)
