from typing import NoReturn

from heavyiq.api.handlers.decorators import with_db, with_feedback_id
from heavyiq.api.models import (
    AnswerResponse,
    AutoQueryResponse,
    AutoQuestionResponse,
    QueryResponse,
    QuestionResponse,
    TablesResponse,
)
from heavyiq.langchain.exceptions import NLtoAnswerException, NLtoSQLException


@with_db
@with_feedback_id
async def handle_lcel_query_request(request_dict: dict, config: dict | None = None) -> QueryResponse | NoReturn:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.lcel.chains import sql_chain
    from heavyiq.lcel.chains.heavydb.sql_chain import max_retries

    out = await sql_chain.ainvoke(request_dict, config=config)  # type: ignore
    if out["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.", failed_sql=out["query"]
        )

    return QueryResponse(sql=out["query"], sql_complexity=out["sql_complexity"], feedback_id="", logprobs={})


@with_db
@with_feedback_id
async def handle_lcel_auto_query_request(
    request_dict: dict, config: dict | None = None
) -> AutoQueryResponse | NoReturn:
    """
    Async LCEL handler for /auto/query request.
    """
    from heavyiq.lcel.chains import auto_sql_chain
    from heavyiq.lcel.chains.heavydb.sql_chain import max_retries

    out = await auto_sql_chain.ainvoke(request_dict, config=config)  # type: ignore
    if out["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.", failed_sql=out["query"]
        )

    return AutoQueryResponse(sql=out["query"], sql_complexity=out["sql_complexity"], feedback_id="", logprobs={})


@with_db
@with_feedback_id
async def handle_lcel_question_request(request_dict: dict, config: dict | None = None) -> QuestionResponse | NoReturn:
    """
    Async LCEL handler for /question request.

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        QuestionResponse
    """
    from heavyiq.lcel.chains import answer_chain
    from heavyiq.lcel.chains.heavydb.sql_chain import max_retries

    result = await answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    fail_reason = result["fail_reason"]

    if fail_reason:
        is_sql_error = True if "SQL Error" in fail_reason else False
        if is_sql_error:
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {max_retries} tries.",
                failed_sql=result["sql"],
            )
        else:
            raise NLtoAnswerException(fail_reason)

    return QuestionResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )


@with_db
@with_feedback_id
async def handle_lcel_auto_question_request(
    request_dict: dict, config: dict | None = None
) -> AutoQuestionResponse | NoReturn:
    """
    Async LCEL handler for /auto/question request.

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        AutoQuestionResponse
    """
    from heavyiq.lcel.chains import auto_answer_chain
    from heavyiq.lcel.chains.heavydb.sql_chain import max_retries

    result = await auto_answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    fail_reason = result["fail_reason"]

    if fail_reason:
        is_sql_error = True if "SQL Error" in fail_reason else False
        if is_sql_error:
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {max_retries} tries.",
                failed_sql=result["sql"],
            )
        else:
            raise NLtoAnswerException(fail_reason)

    return AutoQuestionResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )


@with_db
@with_feedback_id
async def handle_lcel_answer_request(request_dict: dict, config: dict | None = None) -> AnswerResponse | NoReturn:
    """
    Async LCEL handler for /answer request.

    Args:
        request_dict: Input dict with query info (note that the query passed as input should be validated beforehand)
        config: Runnable config dict. Defaults to None.

    Returns:
        AnswerResponse
    """
    from heavyiq.lcel.chains import answer_chain

    result = await answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    if result["fail_reason"]:
        raise NLtoAnswerException(result["fail_reason"])

    return AnswerResponse(
        sql=result["sql"],
        sql_result=result["results"],
        sql_complexity=result["sql_complexity"],
        answer=result["answer"],
        feedback_id="",
    )


@with_db
@with_feedback_id
async def handle_lcel_tables_request(request_dict: dict, config: dict | None = None) -> TablesResponse:
    """
    Async LCEL handler for /tables request (ie, NL to Tables).

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.
    Returns:
        TablesResponse
    """
    from heavyiq.lcel.chains import table_chain

    result = await table_chain.ainvoke(request_dict, config=config)  # type: ignore
    return TablesResponse(tables=result)
