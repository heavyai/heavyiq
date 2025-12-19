"""
Integration tests for RAG retrieve facts chain.

These tests require the embedding server to be running.

Run with:
    pytest tests/heavyrag/chains/test_retrieve_facts_chain.py -v -m integration --config-path config.toml
"""
from typing import TYPE_CHECKING

import pytest

from tests import aoverride_config

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from heavyiq.config import HeavyIQConfig
    from heavyrag.database.base import Database


def populate_vectordb(client: "TestClient", session_id: str):
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
    res = client.post("/rag/snippets/bulk-insert", json=payload)
    assert res.status_code == 200


@pytest.mark.integration
@pytest.mark.anyio
@aoverride_config
async def test_rag_retrieve_snippets_chain_success(
    heavyiq_config: "HeavyIQConfig", client: "TestClient", session_id: str, ragdb: "Database", pre_clean_table
):
    """
    Test heavyrag retrieval of nodes without any re-ranker.
    Requires: embedding server
    """
    from heavyrag.chains.retrieve_facts_chain import chain

    populate_vectordb(client=client, session_id=session_id)
    question = "Which Python libraries are used for numerical computation and data manipulation?"
    expected_answer = (
        "NumPy and Pandas are powerful libraries in Python for numerical computation and data manipulation."
    )

    # # RAG retrieval
    chain_inputs = {
        "question": question,
        "collection_name": heavyiq_config.heavydb_dbname,
        "similarity_cutoff": 0.30,
        "similarity_top_k": 3,
        "use_reranker": False,
        "reranker_cutoff": 0.1,
        "reranker_top_k": 3,
    }
    retrieved_snippets = await chain.ainvoke(chain_inputs)
    assert expected_answer in [i[1] for i in retrieved_snippets]


@pytest.mark.integration
@pytest.mark.anyio
@aoverride_config
async def test_rag_retrieve_and_rerank_snippets(
    heavyiq_config: "HeavyIQConfig", client: "TestClient", session_id: str, ragdb: "Database", pre_clean_table
):
    """
    Test heavyrag retrieval of nodes with re-ranker.
    Requires: embedding server, reranker server
    """
    from heavyrag.chains.retrieve_facts_chain import chain

    populate_vectordb(client=client, session_id=session_id)
    question = "Which Python libraries are used for numerical computation and data manipulation?"
    expected_answer = (
        "NumPy and Pandas are powerful libraries in Python for numerical computation and data manipulation."
    )
    # # RAG retrieval
    chain_inputs = {
        "question": question,
        "collection_name": heavyiq_config.heavydb_dbname,
        "similarity_cutoff": 0.30,
        "similarity_top_k": 2,
        "use_reranker": True,
        "reranker_cutoff": 0.1,
        "reranker_top_k": 1,
    }
    retrieved_snippets = await chain.ainvoke(chain_inputs)
    assert len(retrieved_snippets) == 1
    assert expected_answer == retrieved_snippets[0][1]
