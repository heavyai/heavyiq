# Test RAG controller module (unittests)
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from heavyiq.langchain.heavydb import HeavyDB

pytestmark = pytest.mark.heavydb

if TYPE_CHECKING:
    from heavyrag.controller import BaseController


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


@pytest.mark.anyio
async def test_chroma_vectorstore_passes_facts_insert_and_list(
    chroma_controller_patch: "BaseController", facts_data: list
):
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
async def test_chroma_vectorstore_passes_facts_delete(chroma_controller_patch: "BaseController", facts_data: list):
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
async def test_faiss_vectorstore_passes_facts_insert_and_list(
    faiss_controller_patch: "BaseController", facts_data: list
):
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
async def test_faiss_vectorstore_passes_facts_delete(faiss_controller_patch: "BaseController", facts_data: list):
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


@pytest.mark.anyio
async def test_chroma_vectorstore_passes_tables_sync_and_retrieve(chroma_controller_patch: "BaseController"):
    """
    Test insert tables (fetching table schema, split it into chunks, create embeddings,
    store it inside vectordb with metadata for each node) on chroma vectorstore.
    """
    overture_db_name, cars_db_name = "overture_wa", "cars"
    overture_db = await HeavyDB.from_env_async(db_name=overture_db_name)
    cars_db = await HeavyDB.from_env_async(db_name=cars_db_name)

    await chroma_controller_patch.sync_table_nodes(overture_db, force=True)
    await chroma_controller_patch.sync_table_nodes(cars_db, force=True)

    overture_nodes = await chroma_controller_patch.list_table_nodes(overture_db_name)
    car_nodes = await chroma_controller_patch.list_table_nodes(cars_db_name)

    assert all(n.metadata["type"] == "table" and n.metadata["dbname"] == overture_db_name for n in overture_nodes)
    assert all(n.metadata["type"] == "table" and n.metadata["dbname"] == cars_db_name for n in car_nodes)


@pytest.mark.anyio
async def test_faiss_vectorstore_passes_tables_sync_and_retrieve(faiss_controller_patch: "BaseController"):
    """
    Test insert tables (fetching table schema, split it into chunks, create embeddings,
    store it inside vectordb with metadata for each node) on chroma vectorstore.
    """
    overture_db_name, cars_db_name = "overture_wa", "cars"
    overture_db = await HeavyDB.from_env_async(db_name=overture_db_name)
    cars_db = await HeavyDB.from_env_async(db_name=cars_db_name)

    await faiss_controller_patch.sync_table_nodes(overture_db, force=True)
    await faiss_controller_patch.sync_table_nodes(cars_db, force=True)

    overture_nodes = await faiss_controller_patch.list_table_nodes(overture_db_name)
    car_nodes = await faiss_controller_patch.list_table_nodes(cars_db_name)

    assert all(n.metadata["type"] == "table" and n.metadata["dbname"] == overture_db_name for n in overture_nodes)
    assert all(n.metadata["type"] == "table" and n.metadata["dbname"] == cars_db_name for n in car_nodes)
