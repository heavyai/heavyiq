import functools
from typing import Awaitable, Callable, TypeVar
from uuid import UUID

from langchain import callbacks
from langchain.pydantic_v1 import BaseModel

from heavyiq.api.models import AnswerResponse, QueryResponse, QuestionResponse
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.logging_utils import get_heavyiq_logger

# Define a type variable for the wrapped coroutine function
T = TypeVar("T")


def with_db(coro: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator which wraps the coroutine function within heavydb_context context.
    """

    @functools.wraps(coro)
    async def wrapper(request: BaseModel, db: HeavyDB) -> T:
        config, logger = get_config(), get_heavyiq_logger()
        file_callback_handler = logger.async_langchain_cb_handler(to_stdout=config.log_to_stdout)
        async with heavydb_context(db):  # type: ignore
            return await coro(request.dict(), config={"callbacks": [file_callback_handler]})

    return wrapper


def with_feedback_id(coro: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator which attaches the run_id to the response.
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


@with_db
@with_feedback_id
async def handle_lcel_query_request(request_dict: dict, config: dict | None = None) -> QueryResponse:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.lcel.chains import sql_chain

    out = await sql_chain.ainvoke(request_dict, config=config)  # type: ignore
    return QueryResponse(sql=out["query"], sql_complexity=out["sql_complexity"], feedback_id="", logprobs={})


@with_db
@with_feedback_id
async def handle_lcel_question_request(request_dict: dict, config: dict | None = None) -> QuestionResponse:
    """
    Async LCEL handler for /question request.

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        QuestionResponse
    """
    from heavyiq.lcel.chains import answer_chain

    result = await answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    return QuestionResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )


@with_db
@with_feedback_id
async def handle_lcel_answer_request(request_dict: dict, config: dict | None = None) -> AnswerResponse:
    """
    Async LCEL handler for /answer request.

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        AnswerResponse
    """
    from heavyiq.lcel.chains import answer_chain

    result = await answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    return AnswerResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )
