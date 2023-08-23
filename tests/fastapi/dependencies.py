from typing import Any
from heavyiq.api.models import QueryRequest, GenerateTableMetadataRequest
from heavyiq.langchain import HeavyDB


def override_valid_query_db_session(
    query_request: QueryRequest,
) -> tuple[QueryRequest, Any]:
    return (query_request, HeavyDB.from_env())


def override_valid_table_metadata_db_session(
    query_request: GenerateTableMetadataRequest,
) -> tuple[GenerateTableMetadataRequest, Any]:
    return (query_request, HeavyDB.from_env())
