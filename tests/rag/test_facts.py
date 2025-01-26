import os
from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import patch

# import faiss on test module is must or otherwise we should endup in segmentation fault error upon running tcs
import faiss
import pytest
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient

from heavyiq.config import HeavyIQConfig
from tests import aoverride_config
from tests.api.conftest import client, session_id

if TYPE_CHECKING:
    from heavyrag.database.base import Database


@pytest.fixture(scope="module")
def ragdb(heavyiq_config: HeavyIQConfig) -> Generator["Database", None, None]:

    with patch("heavyrag.database.base.config", heavyiq_config), patch("heavyrag.embed.CONFIG", heavyiq_config):

        from heavyiq.api import rag_initialize
        from heavyrag.database import ragdb

        rag_initialize()
        yield ragdb
        # Extract the file path from the URI
        if ragdb.database_uri.startswith("sqlite:///"):
            file_path = ragdb.database_uri.replace("sqlite:///", "", 1)  # Remove the "sqlite:///" prefix
            # Check if the file exists
            if os.path.exists(file_path):
                # Delete the file
                os.remove(file_path)
                print(f"Database file '{file_path}' has been deleted.")
            else:
                print(f"Database file '{file_path}' does not exist.")
        else:
            print("Invalid SQLite URI format.")


@pytest.fixture(scope="function")
async def pre_clean_table(heavyiq_config: HeavyIQConfig, ragdb: "Database"):
    """
    Fixture which helps to clean/delete all records in facts table and RAG index before running each testcase.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    dbname = heavyiq_config.heavydb_dbname
    with ragdb.get_db() as session:
        FactsModel.delete_facts_by_database(db_session=session, heavydb_name=dbname)

    await rag_controller.delete_fact_nodes(dbname=dbname)
    yield


@pytest.mark.anyio
@aoverride_config
async def test_add_snippet_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test Add snippet `/rag/snippets/insert` endpoint.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    snippet_data = "Use usa_states table when you wish to fetch states related data."
    payload = {"session_id": session_id, "snippet": snippet_data}  # type: ignore
    response = client.post("/rag/snippets/insert", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    snippet_id = response_json["snippet_id"]
    assert snippet_id

    # check for the snippet_id exists on both the ragdb and the vectordb
    # check for fact existence in ragdb (ie. sqlite db)
    with ragdb.get_db() as session:
        fact = FactsModel.get(db_session=session, id=snippet_id)
        assert fact and fact.fact == snippet_data
    nodes = await rag_controller.list_fact_nodes(dbname=fact.heavydb_name)  # type: ignore
    # check for the fact exists in the RAG index
    assert snippet_id in [i.node_id for i in nodes]


@pytest.mark.anyio
@aoverride_config
async def test_bulk_insert_snippet_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test Add snippet `/rag/snippets/bulk-insert` endpoint.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    snippets = [
        "Use the usa_countries table when you need to fetch data related to countries.",
        "Use the ca_tweets table when you need to fetch data related to tweets.",
        "Use the app_store table when you need to fetch data related to apps.",
        "Use usa_states table when you wish to fetch states related data.",
    ]
    payload = {"session_id": session_id, "snippets": snippets}
    response = client.post("/rag/snippets/bulk-insert", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    snippet_ids = response_json["snippet_ids"]
    assert snippet_ids

    # check for facts existence in ragdb (ie. sqlite db)
    with ragdb.get_db() as session:
        facts = FactsModel.list(db_session=session, heavydb_name=heavyiq_config.heavydb_dbname)
        assert all([i.id in snippet_ids for i in facts])

    nodes = await rag_controller.list_fact_nodes(dbname=heavyiq_config.heavydb_dbname)  # type: ignore
    # check for the facts exists in the RAG index
    assert all([i.node_id in snippet_ids for i in nodes])


@pytest.mark.anyio
@aoverride_config
async def test_update_snippet_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test Update snippet `/rag/snippets/update` endpoint.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    # add snippet
    snippet_data = "Use usa_states table when you wish to fetch states related data."
    payload = {"session_id": session_id, "snippet": snippet_data}  # type: ignore
    response = client.post("/rag/snippets/insert", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    snippet_id = response_json["snippet_id"]

    new_snippet = "Use the usa_countries table when you need to fetch data related to countries."
    payload = {"session_id": session_id, "snippet_id": snippet_id, "snippet": new_snippet}
    response = client.post("/rag/snippets/update", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    updated_snippet_id = response_json["snippet_id"]
    assert snippet_id == updated_snippet_id

    # check for facts existence in ragdb (ie. sqlite db)
    with ragdb.get_db() as session:
        fact = FactsModel.get(db_session=session, id=snippet_id)
        assert fact and fact.fact == new_snippet
    nodes = await rag_controller.list_fact_nodes(dbname=fact.heavydb_name)  # type: ignore
    # check for the fact exists in the RAG index
    found_node = [node for node in nodes if node.node_id == snippet_id]
    assert found_node
    assert found_node[0].text == new_snippet


@pytest.mark.anyio
@aoverride_config
async def test_get_snippet_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test GET snippet `/rag/snippets/get` endpoint.
    """
    # add snippet
    snippet_data = "Use usa_states table when you wish to fetch states related data."
    payload = {"session_id": session_id, "snippet": snippet_data}  # type: ignore
    response = client.post("/rag/snippets/insert", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    snippet_id = response_json["snippet_id"]

    # get snippet
    payload: dict["str", "str" | bool] = {"session_id": session_id, "snippet_id": snippet_id, "verify": True}
    response = client.post("/rag/snippets/get", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["snippet"] == snippet_data
    assert response_json["snippet_id"] == snippet_id

    # should raise 404 error on snipped not found
    modified_snippet_id = snippet_id[:-4] + "aaaa"
    payload: dict["str", "str" | bool] = {
        "session_id": session_id,
        "snippet_id": modified_snippet_id,
        "verify": True,
    }
    response = client.post("/rag/snippets/get", json=payload)
    assert response.status_code == 500


@pytest.mark.anyio
@aoverride_config
async def test_delete_snippets_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test Delete snippet or snippets using `/rag/snippets/delete` endpoint.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    # add snippets
    snippets = [
        "Use the usa_countries table when you need to fetch data related to countries.",
        "Use the ca_tweets table when you need to fetch data related to tweets.",
        "Use the app_store table when you need to fetch data related to apps.",
        "Use usa_states table when you wish to fetch states related data.",
    ]
    payload = {"session_id": session_id, "snippets": snippets}
    response = client.post("/rag/snippets/bulk-insert", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    snippet_ids = response_json["snippet_ids"]
    assert snippet_ids

    # check for the existence of fact ids in ragdb and vector index
    with ragdb.get_db() as session:
        assert FactsModel.get(db_session=session, id=snippet_ids[0])

    nodes = await rag_controller.list_fact_nodes(dbname=heavyiq_config.heavydb_dbname)  # type: ignore
    # check for the facts exists in the RAG index
    node_ids = [i.node_id for i in nodes]
    assert snippet_ids[0] in node_ids

    # delete snippets
    payload = {"session_id": session_id, "snippet_ids": snippet_ids}  # type: ignore
    response = client.post("/rag/snippets/delete", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["deleted"] == True

    # check for the existence of fact ids
    with ragdb.get_db() as session:
        fact = FactsModel.get(db_session=session, id=snippet_ids[0])
        assert fact is None

    nodes = await rag_controller.list_fact_nodes(dbname=heavyiq_config.heavydb_dbname)  # type: ignore
    # check for the facts exists in the RAG index
    node_ids = [i.node_id for i in nodes]
    assert snippet_ids[0] not in node_ids


@pytest.mark.anyio
@aoverride_config
async def test_sync_snippets_success(
    heavyiq_config: HeavyIQConfig, pre_clean_table: None, ragdb: "Database", client: TestClient, session_id: str
):
    """
    Test Sync snippets through `/rag/snippets/sync` endpoint.
    ie. Makes sure the snippet exists on ragdb should have the relevant entry/node on the vectordb ("chroma", "faiss) index.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    # create an entry on Facts table, run sync and finally check for relevant node existence in vectorstore.
    fact, fact_id = "Can you find this fact on vectorstore?", None
    with ragdb.get_db() as session:
        fact_id = FactsModel.add(db_session=session, heavydb_name=heavyiq_config.heavydb_dbname, fact=fact)
    assert fact_id

    payload = {"session_id": session_id, "force": False}  # type: ignore
    response = client.post("/rag/snippets/sync", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["success"] is True

    nodes = await rag_controller.list_fact_nodes(dbname=heavyiq_config.heavydb_dbname)  # type: ignore
    # check for the facts exists in the RAG index
    texts = [i.text for i in nodes]
    assert fact in texts


@pytest.mark.anyio
@aoverride_config
async def test_search_snippets_success(
    heavyiq_config: HeavyIQConfig,
    pre_clean_table: None,
    ragdb: "Database",
    client: TestClient,
    session_id: str,
):
    """
    Test Search snippets through `/rag/snippets/search` endpoint.
    """
    # from heavyrag.vector_stores.faiss import FaissIQVectorStore

    # create an entry on Facts table
    snippets = [
        "Python is a versatile programming language used widely for web development, data analysis, and machine learning.",
        "Django is a popular Python framework for building secure and scalable web applications.",
        "FastAPI is a modern, fast web framework for building APIs with Python, leveraging asynchronous capabilities.",
        "NumPy and Pandas are powerful libraries in Python for numerical computation and data manipulation.",
        "SQLAlchemy is an ORM library in Python that simplifies database operations.",
        "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
        "Machine learning models can be developed using libraries like Scikit-learn and TensorFlow in Python.",
        "Python's standard library provides extensive functionality, reducing the need for third-party dependencies.",
        "Testing in Python is streamlined with frameworks like pytest and unittest.",
        "Jupyter Notebooks are widely used for interactive data analysis and visualization in Python.",
    ]
    payload = {"session_id": session_id, "snippets": snippets}
    response = client.post("/rag/snippets/bulk-insert", json=payload)
    assert response.status_code == 200

    question = "Which Python libraries are used for numerical computation and data manipulation?"
    expected_answer = (
        "NumPy and Pandas are powerful libraries in Python for numerical computation and data manipulation."
    )
    payload: dict[str, str | int] = {"session_id": session_id, "question": question, "top_k": 2}
    response = client.post("/rag/snippets/search", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert expected_answer in [i["content"] for i in response_json["nodes"]]
