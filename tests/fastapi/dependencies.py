from typing import Any
from heavyiq.api.models import QueryRequest
from heavyiq.langchain import HeavyDB


def override_valid_query_db_session(
    query_request: QueryRequest,
) -> tuple[QueryRequest, Any]:
    return (query_request, HeavyDB.from_env())
