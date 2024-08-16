from typing import NoReturn
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from heavyiq.api.handlers.decorators import get_logging_callback, with_db, with_feedback_id
from heavyiq.api.models import (
    AnswerResponse,
    AutoQueryRequest,
    AutoQueryResponse,
    AutoQuestionRequest,
    AutoQuestionResponse,
    COTQueryResponse,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
    TablesResponse,
    TablesToQuestionsResponse,
)
from heavyiq.config import get_config
from heavyiq.langchain.exceptions import NLtoAnswerException, NLtoSQLException, NLtoTableException

callbacks: list[BaseCallbackHandler] = [get_logging_callback()]


@with_feedback_id
async def handle_graph_query_request(request: QueryRequest) -> QueryResponse | NoReturn:
    """
    Async graph handler for /graph/query request.
    """
    from heavyiq.langgraph.graphs.sql_graph import graph

    global_config = get_config()

    max_retries, is_logprobs_enabled, logprobs, total_score = (
        global_config.max_retries_nl_to_sql,
        global_config.enable_logprobs,
        {},
        None,
    )
    config: RunnableConfig = {
        "configurable": {"session_id": request.session_id, "do_generate_answer": False, "thread_id": request.thread_id},
        "callbacks": callbacks,
    }
    response = await graph.ainvoke(request.dict(), config=config)
    if response["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.",
            failed_sql=response["query"],
        )

    return QueryResponse(
        sql=response["query"],
        sql_complexity=response["sql_complexity"],
        feedback_id="",
        logprobs=logprobs,
        total_score=total_score,
        snippet_ids=response.get("snippet_ids", []),
        thread_id=request.thread_id,
    )


@with_feedback_id
async def handle_graph_auto_query_request(request: AutoQueryRequest) -> AutoQueryResponse | NoReturn:
    """
    Async graph handler for /graph/auto/query request.
    """
    from heavyiq.langgraph.graphs.sql_graph import graph

    max_retries = get_config().max_retries_nl_to_sql
    config: RunnableConfig = {
        "configurable": {"session_id": request.session_id, "do_generate_answer": False, "thread_id": request.thread_id},
        "callbacks": callbacks,
    }

    out = await graph.ainvoke(request.dict(), config=config)  # type: ignore

    if not out["tables"]:
        raise NLtoTableException("No relevant tables found to answer question.")

    if out["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.", failed_sql=out["query"]
        )

    return AutoQueryResponse(
        sql=out["query"],
        sql_complexity=out["sql_complexity"],
        feedback_id="",
        tables=out["tables"],
        logprobs={},
        snippet_ids=out["snippet_ids"],
        thread_id=request.thread_id,
    )


@with_feedback_id
async def handle_graph_question_request(request: QuestionRequest) -> QuestionResponse | NoReturn:
    """
    Async graph handler for /graph/question request.
    """
    from heavyiq.langgraph.graphs.sql_graph import graph

    max_retries = get_config().max_retries_nl_to_sql
    config: RunnableConfig = {
        "configurable": {"session_id": request.session_id, "do_generate_answer": True, "thread_id": request.thread_id},
        "callbacks": callbacks,
    }

    result = await graph.ainvoke(request, config=config)  # type: ignore
    fail_reason = result["error"]

    if fail_reason:
        is_sql_error = True if "SQL Error" in fail_reason else False
        if is_sql_error:
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {max_retries} tries.",
                failed_sql=result["query"],
            )
        else:
            raise NLtoAnswerException(fail_reason)

    return QuestionResponse(
        sql=result["query"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
        snippet_ids=result["snippet_ids"],
        thread_id=request.thread_id,
    )


@with_feedback_id
async def handle_graph_auto_question_request(request: AutoQuestionRequest) -> AutoQuestionResponse | NoReturn:
    """
    Async graph handler for /graph/auto/question request.
    """
    from heavyiq.langgraph.graphs.sql_graph import graph

    max_retries = get_config().max_retries_nl_to_sql
    config: RunnableConfig = {
        "configurable": {"session_id": request.session_id, "do_generate_answer": True, "thread_id": request.thread_id},
        "callbacks": callbacks,
    }

    result = await graph.ainvoke(request, config=config)

    result = await graph.ainvoke(request, config=config)  # type: ignore
    if not result["tables"]:
        raise NLtoTableException("No relevant tables found to answer question.")

    fail_reason = result["error"]

    if fail_reason:
        is_sql_error = True if "SQL Error" in fail_reason else False
        if is_sql_error:
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {max_retries} tries.",
                failed_sql=result["query"],
            )
        else:
            raise NLtoAnswerException(fail_reason)

    return AutoQuestionResponse(
        sql=result["query"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
        tables=result["tables"],
        snippet_ids=result["snippet_ids"],
        thread_id=request.thread_id,
    )
