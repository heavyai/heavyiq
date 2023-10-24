import re
import asyncio
import json
from typing import Any, Iterable, Iterator, Callable
from collections.abc import Awaitable
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class AsyncIteratorWrapper:
    """The following is a utility class that transforms a
    regular iterable to an asynchronous one.

    link: https://www.python.org/dev/peps/pep-0492/#example-2
    """

    _it: Iterator[Any]

    def __init__(self, obj: Iterable[Any]) -> None:
        self._it = iter(obj)

    def __aiter__(self) -> "AsyncIteratorWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        return await fn
    except Exception as e:
        # TODO: handle exception
        print(f"Caught exception: {e}")
        raise e
    finally:
        # Signal the aiter to stop.
        event.set()


class NDJsonStreamingResponse(StreamingResponse):
    """
    This converts the text events received into newline delimited json
    and then streams ndjson string as response.
    """

    media_type = "application/x-ndjson"
    PARSE_CHUNK_RGX = re.compile(r"(?s)event: (.*?)\ndata: (.*?)\n\n")

    async def _chunk_to_json(self, chunk: str) -> str:
        """
        Convert the text chunk to nd json chunk.
        "event: event_name\ndata: message\n\n" -> {"event": "event_name", "data": "message"}\n
        """
        match = self.PARSE_CHUNK_RGX.search(chunk)
        assert match
        return json.dumps({"event": match.group(1), "data": match.group(2)}) + "\n"

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
                json_chunk = await self._chunk_to_json(chunk)
                chunk = json_chunk.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        await send({"type": "http.response.body", "body": b"", "more_body": False})
