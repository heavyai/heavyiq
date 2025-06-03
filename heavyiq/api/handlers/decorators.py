import functools
from typing import Awaitable, Callable, TypeVar
from uuid import UUID

from langchain import callbacks
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic import BaseModel

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.lcel.callbacks.file_callback import LogFileCallbackHandler
from heavyiq.logging_utils import get_heavyiq_logger

# Define a type variable for the wrapped coroutine function
T = TypeVar("T")


def with_db(coro: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator for LCEL handler which wraps the coroutine function within heavydb_context context.
    """

    @functools.wraps(coro)
    async def wrapper(request: BaseModel, db: HeavyDB) -> T:
        logger = get_heavyiq_logger()
        async with heavydb_context(db):  # type: ignore
            try:
                request_dict = request if isinstance(request, dict) else request.dict()
                response = await coro(request_dict, config={"callbacks": [LogFileCallbackHandler(logger=logger)]})
            except Exception as e:
                raise e
            else:
                return response
            finally:
                wait_for_all_tracers()

    return wrapper


def with_feedback_id(coro: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator for LCEL handler which attaches the run_id to the response.
    """

    async def wrapper(*args, **kwargs) -> T:
        from heavyiq.langchain.utils import is_langsmith_active

        run_id: UUID | str = ""
        if is_langsmith_active:
            with callbacks.collect_runs() as cb:
                out = await coro(*args, **kwargs)
                run_id = str(cb.traced_runs[0].id)
        else:
            out = await coro(*args, **kwargs)
        if isinstance(out, BaseModel) and hasattr(out, "feedback_id"):
            setattr(out, "feedback_id", run_id)

        return out

    return wrapper
