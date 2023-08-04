import pytest
from typing import Any, Callable
from .base import client
from unittest.mock import patch

SQL = "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;"


# Mock the log_chain_call_async function to return "foo"
async def mock_log_chain_call_async(chain: Any, chain_input: dict, model_name: str) -> dict[str, str]:
    return {"sql": SQL}


@pytest.mark.parametrize("expected_result", [{"sql": SQL, "sql_complexity": 3}])
@patch("heavyiq.fastapi.handlers.iq_handler.log_chain_call_async", side_effect=mock_log_chain_call_async)
def test_should_pass_query_endpoint(mock: Callable, expected_result: list[Any]):
    """
    Test query endpoint.
    """

    payload = {
        "session_id": "x" * 32,
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["usa_states"],
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    assert response.json() == expected_result
