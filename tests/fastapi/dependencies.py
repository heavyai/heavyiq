from typing import Any

import pytest

from heavynl.fastapi.models import QueryRequest
from heavynl.langchain import HeavyDB


@pytest.fixture
def mock_database(mocker: Any) -> Any:
    """
    HeavyDB mock object
    """
    return mocker.MagicMock()


def override_valid_query_db_session(
    query_request: QueryRequest,
) -> tuple[QueryRequest, Any]:
    return (query_request, HeavyDB.from_env())
