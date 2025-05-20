from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient


@patch("heavyiq.langchain.utils.is_langsmith_active", False)
def test_should_fail_upon_submitting_feedback(client: TestClient):
    """
    Test /submit-feedback endpoint.
    """
    payload = {
        "feedback_id": "a524",
        "score": 1.0,
        "comment": "This produced a geo-spatial join that was very helpful.",
    }
    response = client.post("/api/v1/submit-feedback", json=payload)
    assert response.status_code == 500
    assert response.json() == {"error": "HTTP Exception: Langsmith is not enabled in the config."}


@patch("heavyiq.langchain.utils.is_langsmith_active", True)
@patch("heavyiq.api.handlers.lcel_handler.Client")
def test_should_pass_upon_submitting_feedback(MockClient: Any, client: TestClient):
    """
    Test /submit-feedback endpoint.
    """
    payload = {
        "feedback_id": "a524",
        "score": 1.0,
        "comment": "This produced a geo-spatial join that was very helpful.",
    }
    mock_instance = MockClient.return_value
    mock_instance.create_feedback.return_value = True
    response = client.post("/api/v1/submit-feedback", json=payload)
    assert response.status_code == 200
