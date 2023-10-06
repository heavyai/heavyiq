import pytest
from typing import Any, Callable
from tests.api import RunId
from unittest.mock import patch

SQL = "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;"
ANSWER = "4 states start with the letter A; Alaska, Arizona, Arkansas, and Alabama."


# Mock the log_chain_call_async function to return "foo"
async def mock_log_chain_call_async(chain: Any, chain_input: dict, model_name: str) -> dict[str, str | int | RunId]:
    return {
        "sql": SQL,
        "answer": ANSWER,
        "sql_complexity": 3,
        "__run": RunId("acf2132"),
        "results": """[(1, "Alaska"), (1, "Arizona"), (1, "Arkansas"), (1, "Alabama")]""",
        "info": "",
    }


async def mock_log_chain_call_async_with_fail(
    chain: Any, chain_input: dict, model_name: str
) -> dict[str, str | int | RunId]:
    return {
        "sql": SQL,
        "answer": "",
        "sql_complexity": 3,
        "__run": RunId("acf2132"),
        "results": """[(1, "Alaska"), (1, "Arizona"), (1, "Arkansas"), (1, "Alabama")]""",
        "info": "Generated SQL query resultset exceeds the defined maximum result set size.",
    }


@pytest.mark.parametrize(
    "expected_result",
    [
        [
            {
                "sql": SQL,
                "answer": ANSWER,
                "sql_complexity": 3,
                "feedback_id": "acf2132",
                "sql_result": """[(1, "Alaska"), (1, "Arizona"), (1, "Arkansas"), (1, "Alabama")]""",
            },
            {
                "sql": SQL,
                "answer": "Generated SQL query resultset exceeds the defined maximum result set size.",
                "sql_complexity": 3,
                "feedback_id": "acf2132",
                "sql_result": """[(1, "Alaska"), (1, "Arizona"), (1, "Arkansas"), (1, "Alabama")]""",
            },
        ]
    ],
)
@patch("heavyiq.api.handlers.iq_handler.log_chain_call_async", side_effect=mock_log_chain_call_async)
def test_should_pass_question_endpoint(mock: Callable, expected_result: list[Any], client):
    """
    Test query endpoint.
    """

    payload = {
        "session_id": "x" * 32,  # type: ignore
        "question": "How many states begin with the letter A? What are they?",
        "tables": ["usa_states"],
    }
    response = client.post("/api/v1/question", json=payload)
    assert response.status_code == 200
    assert response.json() == expected_result[0]  # type: ignore
    with patch("heavyiq.api.handlers.iq_handler.log_chain_call_async", side_effect=mock_log_chain_call_async_with_fail):
        response = client.post("/api/v1/question", json=payload)
        assert response.status_code == 200
        assert response.json() == expected_result[1]  # type: ignore
