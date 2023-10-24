from starlette.responses import AsyncContentStream
from fastapi import APIRouter, Depends, Response
from heavyiq.langchain import HeavyDB
from heavyiq.api.dependencies import (
    valid_query_db_session,
)
from heavyiq.api.models import QueryRequest
from fastapi.responses import StreamingResponse
from heavyiq.api.handlers import (
    streaming_query,
)
from heavyiq.api.utils import NDJsonStreamingResponse

streamrouter = APIRouter()
stream_token_router = APIRouter()


async def text_stream(stream_coro: AsyncContentStream) -> StreamingResponse:
    """
    Wrapper function which helps to stream the output yielded from stream_coroutine as text.
    """
    response = StreamingResponse(stream_coro, media_type="text/event-stream")
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


async def ndjson_stream(stream_coro: AsyncContentStream) -> NDJsonStreamingResponse:
    """
    Wrapper function which helps to stream the output yielded from stream_coroutine as nd-json.
    """
    response = NDJsonStreamingResponse(stream_coro)
    response.headers["Content-Type"] = "application/x-ndjson"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


@streamrouter.post("/query", response_class=Response)
async def streaming_query_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    return await ndjson_stream(streaming_query(*values, stream_llm_tokens=False))


@stream_token_router.post("/query", response_class=Response)
async def streaming_query_with_token_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    return await text_stream(streaming_query(*values, stream_llm_tokens=True))


streamrouter.include_router(stream_token_router, prefix="/token")
