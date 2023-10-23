import asyncio
from typing import Any, Iterable, Iterator
from collections.abc import Awaitable


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
