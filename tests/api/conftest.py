from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from heavyiq.api import create_app
from heavyiq.api.dependencies import (
    valid_query_db_session,
    valid_question_db_session,
    validate_db_session_for_table_metadata,
)
from heavyiq.config import HeavyIQConfig
from heavyiq.langchain import HeavyDB

from .dependencies import (
    override_heavydb_client_for_query_request,
    override_heavydb_client_for_question_request,
    override_valid_table_metadata_db_session,
)


@pytest.fixture(scope="package")
def client(config_file_path):
    # Initialize the TestClient with the provided base URL
    app = create_app(config_file_path)
    app.dependency_overrides[valid_query_db_session] = override_heavydb_client_for_query_request
    app.dependency_overrides[valid_question_db_session] = override_heavydb_client_for_question_request
    app.dependency_overrides[validate_db_session_for_table_metadata] = override_valid_table_metadata_db_session
    return TestClient(app)


@pytest.fixture(scope="package")
def session_id(heavyiq_config: HeavyIQConfig):
    try:
        # Use patch to mock the get_config function
        with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
            yield HeavyDB.create_session_id()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None
