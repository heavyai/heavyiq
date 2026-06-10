# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import TypedDict

from fastapi.responses import StreamingResponse
from starlette.responses import AsyncContentStream
from starlette.types import Send


class NDJsonChunkType(TypedDict):
    event: str
    data: str


class NDJsonStreamingResponse(StreamingResponse):
    """
    This converts the text events received into newline delimited json
    and then streams ndjson string as response.
    """

    media_type = "application/x-ndjson"
    PARSE_CHUNK_RGX = re.compile(r"(?s)event: (.*?)\ndata: (.*?)\n\n")

    async def _text_chunk_to_json(self, chunk: str) -> str:
        """
        Converts the text chunk to nd json chunk.
        "event: event_name\ndata: message\n\n" -> {"event": "event_name", "data": "message"}\n
        """
        match = self.PARSE_CHUNK_RGX.search(chunk)
        assert match
        return json.dumps({"event": match.group(1), "data": match.group(2)}) + "\n"

    async def _dict_chunk_to_json(self, chunk: NDJsonChunkType) -> str:
        """
        Converst the dict chunk to nd json chunk.
        """
        return json.dumps(chunk) + "\n"

    async def stream_response(self, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        async for chunk in self.body_iterator:
            if isinstance(chunk, str):
                json_chunk = await self._text_chunk_to_json(chunk)
                chunk = json_chunk.encode(self.charset)
            elif isinstance(chunk, dict):
                json_chunk = await self._dict_chunk_to_json(chunk)
                chunk = json_chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def ndjson_stream(stream_coro: AsyncContentStream) -> NDJsonStreamingResponse:
    """
    Wrapper function which helps to stream the output yielded from stream_coroutine as nd-json.
    """
    response = NDJsonStreamingResponse(stream_coro)
    response.headers["Content-Type"] = "application/x-ndjson"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response
