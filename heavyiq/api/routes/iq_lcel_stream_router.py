from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from heavyiq.api.dependencies import valid_query_db_session, valid_question_db_session
from heavyiq.api.models import QueryRequest, QuestionRequest
from heavyiq.api.responses import ndjson_stream
from heavyiq.langchain import HeavyDB

streamrouter = APIRouter()


@streamrouter.post("/query", response_class=Response)
async def streaming_query_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    from heavyiq.api.handlers.lcel_stream_handler import streaming_query

    return await ndjson_stream(streaming_query(*values))  # type: ignore


@streamrouter.post("/cot_query", response_class=Response)
async def streaming_query_with_cot__endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    from heavyiq.api.handlers.lcel_stream_handler import streaming_query_with_cot

    return await ndjson_stream(streaming_query_with_cot(*values))  # type: ignore


@streamrouter.post("/question", response_class=Response)
async def streaming_question_endpoint(
    values: tuple[QuestionRequest, HeavyDB] = Depends(valid_question_db_session)
) -> StreamingResponse:
    from heavyiq.api.handlers.lcel_stream_handler import streaming_question

    return await ndjson_stream(streaming_question(*values))  # type: ignore
