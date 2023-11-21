from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from heavyiq.api.dependencies import valid_query_db_session
from heavyiq.api.models import QueryRequest
from heavyiq.api.responses import ndjson_stream
from heavyiq.langchain import HeavyDB

streamrouter = APIRouter()


@streamrouter.post("/query", response_class=Response)
async def streaming_query_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    from heavyiq.api.handlers.lcel_stream_handler import streaming_query

    return await ndjson_stream(streaming_query(*values))  # type: ignore
