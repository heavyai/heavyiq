from pydantic import BaseModel
from typing import Coroutine
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

streamrouter = APIRouter()
stream_token_router = APIRouter()


async def stream(stream_coro: AsyncContentStream) -> StreamingResponse:
    """
    Wrapper function which helps to stream the output yielded from stream_coroutine.
    """
    response = StreamingResponse(stream_coro, media_type="text/event-stream")
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response


@streamrouter.post("/query", response_class=Response)
async def streaming_query_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    return await stream(streaming_query(*values, stream_llm_tokens=False))


@stream_token_router.post("/query", response_class=Response)
async def streaming_query_with_token_endpoint(
    values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)
) -> StreamingResponse:
    return await stream(streaming_query(*values, stream_llm_tokens=True))


streamrouter.include_router(stream_token_router, prefix="/token")
