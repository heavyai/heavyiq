import pytest
from typing import Any, Callable
from tests.api import RunId
from unittest.mock import patch


# Mock the log_chain_call_async function
async def mock_log_chain_call_async(chain: Any, chain_input: dict, model_name: str) -> dict[str, str | dict | RunId]:
    return {
        "summary": 'Description:\nThe "us_pois_safegraph" table is designed to store information about points of interest (POIs) in the United States.',
        "columns": {
            "safegraph_place_id": "A unique identifier for each place in the dataset.",
            "parent_safegraph_place_id": "The safegraph_place_id of the parent place, if applicable.",
            "location_name": "The name of the location.",
            "brands": "The name of the brand associated with the location.",
        },
        "__run": RunId("acf2132"),
    }


@pytest.mark.parametrize(
    "expected_result",
    [
        {
            "table_name": "us_pois_safegraph",
            "summary": 'Description:\nThe "us_pois_safegraph" table is designed to store information about points of interest (POIs) in the United States.',
            "columns": {
                "safegraph_place_id": "A unique identifier for each place in the dataset.",
                "parent_safegraph_place_id": "The safegraph_place_id of the parent place, if applicable.",
                "location_name": "The name of the location.",
                "brands": "The name of the brand associated with the location.",
            },
            "feedback_id": "acf2132",
        },
    ],
)
@patch("heavyiq.api.handlers.iq_handler.log_chain_call_async", side_effect=mock_log_chain_call_async)
def test_should_pass_for_generate_table_metadata_endpoint(mock: Callable, expected_result: dict[str, Any], client):
    """
    Test /generate-table-metadata endpoint.
    """

    payload = {
        "session_id": "x" * 32,  # type: ignore
        "table_name": "us_pois_safegraph",
    }
    response = client.post("/api/v1/generate-table-metadata", json=payload)
    assert response.status_code == 200
    assert response.json() == expected_result
