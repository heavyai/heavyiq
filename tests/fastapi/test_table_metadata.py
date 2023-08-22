import pytest
from typing import Any, Callable
from .base import client
from unittest.mock import patch
from heavyiq.langchain.heavydb import HeavyDB


# async_create_table_metadata mock
async def mock_async_create_table_metadata(db: HeavyDB, table_name: str) -> str:
    """
    Mock function.
    """
    table_summary = """Description:\nThe "us_pois_safegraph" table is designed to store information about points of interest (POIs) in the United States."""
    column_description = """
        Column descriptions for us_pois_safegraph table:
            - us_pois_safegraph.safegraph_place_id: A unique identifier for each place in the dataset.
            - us_pois_safegraph.parent_safegraph_place_id: The safegraph_place_id of the parent place, if applicable.
            - us_pois_safegraph.location_name: The name of the location.
            - us_pois_safegraph.brands: The name of the brand associated with the location.
    """
    return f"{table_summary}\n{column_description}"


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
        },
    ],
)
@patch("heavyiq.api.handlers.iq_handler.async_create_table_metadata", side_effect=mock_async_create_table_metadata)
def test_should_pass_for_generate_table_metadata_endpoint(mock: Callable, expected_result: dict[str, Any]):
    """
    Test /generate-table-metadata endpoint.
    """

    payload = {
        "session_id": "x" * 32,
        "table_name": "us_pois_safegraph",
    }
    response = client.post("/api/v1/generate-table-metadata", json=payload)
    assert response.status_code == 200
    assert response.json() == expected_result
