# Test LCEL query endpoints (integration tests)

import pytest
from fastapi.testclient import TestClient

from heavyiq.config import HeavyIQConfig
from tests import aoverride_config, override_config


@pytest.mark.anyio
@aoverride_config
async def test_should_pass_lcel_query_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
    """
    Test LCEL query endpoint.
    """

    payload = {
        "session_id": session_id,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["heavyai_us_states"],
    }
    response = client.post("/api/v1/lcel/query", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["sql"]
    assert response_json["sql_complexity"] in [2, 3]


@pytest.mark.anyio
@aoverride_config
async def test_should_pass_lcel_question_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
    """
    Test LCEL question endpoint.
    """

    payload = {
        "session_id": session_id,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["heavyai_us_states"],
    }
    response = client.post("/api/v1/lcel/question", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["answer"]
    assert response_json["sql"]
    assert response_json["sql_result"]
    assert response_json["sql_complexity"] in [2, 3]


@pytest.mark.anyio
@aoverride_config
async def test_should_pass_lcel_tables_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
    """
    Test LCEL tables endpoint.
    """

    payload = {
        "session_id": session_id,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "allowed_tables": ["heavyai_us_states", "heavyai_countries"],
    }
    response = client.post("/api/v1/lcel/tables", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["tables"]["heavyai_us_states"] == 1
    assert response_json["tables"]["heavyai_countries"] == 0


@pytest.mark.anyio
@aoverride_config
async def test_should_pass_lcel_auto_query_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
    """
    Test LCEL auto table query endpoint.
    """
    payload = {
        "session_id": session_id,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "allowed_tables": ["heavyai_us_states", "heavyai_countries"],
    }
    response = client.post("/api/v1/lcel/auto/query", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["sql"]
    assert response_json["sql_complexity"] in [2, 3]


@pytest.mark.anyio
@aoverride_config
async def test_should_pass_lcel_auto_question_endpoint(
    heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str
):
    """
    Test LCEL auto table question endpoint.
    """
    assert session_id, "Invalid session"
    payload = {
        "session_id": session_id,
        "question": "How many states begin with the letter A? What are they?",
        "allowed_tables": ["heavyai_us_states", "heavyai_countries"],
    }
    response = client.post("/api/v1/lcel/auto/question", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["answer"]
    assert response_json["sql"]
    assert response_json["sql_result"]
    assert response_json["sql_complexity"] in [2, 3]


@pytest.mark.anyio
@override_config
async def test_should_pass_lcel_answer_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
    """
    Test LCEL answer endpoint.
    """

    payload = {
        "session_id": session_id,
        "query": "SELECT COUNT(DISTINCT STATE_NAME) AS count_states, STATE_NAME FROM heavyai_us_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["heavyai_us_states"],
    }
    response = client.post("/api/v1/lcel/answer", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["answer"]
    assert response_json["sql"]
    assert response_json["sql_result"]
    assert response_json["sql_complexity"] in [2, 3]


@pytest.mark.anyio
@override_config
async def test_should_pass_call_llm_endpoint(heavyiq_config: HeavyIQConfig, client: TestClient):
    """
    Test LCEL answer endpoint.
    """

    payload = {
        "question": "Return only the name of the capital for the following state, Idaho",
        "temperature": 0.0,
        "max_tokens": 256,
        "stop": ["is the capital"],
    }
    response = client.post("/llm/call-llm", json=payload)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["response"] == "Boise"
