from typing import Any
from .base import client
from unittest.mock import patch


@patch("heavyiq.langchain.utils.is_langsmith_active", False)
def test_should_fail_upon_submitting_feedback():
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
    assert response.json() == {"error": "HTTP Exception: Feedback is not enabled in the config."}


@patch("heavyiq.langchain.utils.is_langsmith_active", True)
@patch("heavyiq.api.handlers.iq_handler.Client")
def test_should_pass_upon_submitting_feedback(MockClient: Any):
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
