import functools
from typing import Awaitable, Callable, TypeVar

from langchain.pydantic_v1 import BaseModel

from heavyiq.api.models import QueryRequest, QueryResponse, QuestionRequest, QuestionResponse
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


@with_db
async def handle_lcel_query_request(request_dict: dict, config: dict | None = None) -> QueryResponse:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.controller.langchain import predict_sql_query_and_complexity_with_retries

    out = await predict_sql_query_and_complexity_with_retries(request_dict, config=config)  # type: ignore
    return QueryResponse(sql=out[0], sql_complexity=out[1], feedback_id="", logprobs={})


@with_db
async def handle_lcel_question_request(request_dict: dict, config: dict | None = None) -> QuestionResponse:
    """
    Async LCEL handler for /question request.

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        QuestionResponse
    """
    from heavyiq.controller.langchain import predict_sql_query_and_answer_with_retries

    result = await predict_sql_query_and_answer_with_retries(request_dict, config=config)  # type: ignore
    return QuestionResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )
