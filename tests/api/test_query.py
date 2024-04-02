from typing import Any, Callable
from unittest.mock import patch

import pytest

from tests.api import RunId

SQL = "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;"


# Mock the log_chain_call_async function to return "foo"
async def mock_log_chain_call_async(
    chain: Any, chain_input: dict, model_name: str
) -> dict[str, str | int | RunId | dict]:
    return {"sql": SQL, "sql_complexity": 3, "__run": RunId("acf2132"), "logprobs": {"token_logprobs": []}}


@pytest.mark.parametrize(
    "expected_result",
    [
        {
            "sql": SQL,
            "sql_complexity": 3,
            "feedback_id": "acf2132",
            "logprobs": {"token_logprobs": []},
            "total_score": None,
        }
    ],
)
@patch("heavyiq.api.handlers.iq_handler.log_chain_call_async", side_effect=mock_log_chain_call_async)
def test_should_pass_query_endpoint(mock: Callable, expected_result: list[Any], client):
    """
    Test query endpoint.
    """

    payload = {
        "session_id": "x" * 32,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["usa_states"],
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    assert response.json() == expected_result
