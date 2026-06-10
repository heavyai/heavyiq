# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn

from fastapi import HTTPException
from langsmith import Client

from heavyiq.api.handlers.decorators import with_db, with_feedback_id
from heavyiq.api.handlers.utils import ainvoke_logprobs, compute_total_probability
from heavyiq.api.models import (
    AnswerResponse,
    AutoQueryResponse,
    AutoQuestionResponse,
    COTQueryResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryResponse,
    QuestionResponse,
    TablesResponse,
    TablesToQuestionsResponse,
)
from heavyiq.config import get_config
from heavyiq.langchain.exceptions import NLtoAnswerException, NLtoSQLException, NLtoTableException


@with_db
@with_feedback_id
async def handle_lcel_query_request(request_dict: dict, config: dict | None = None) -> QueryResponse | NoReturn:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.lcel.chains import sql_chain

    global_config = get_config()

    max_retries, is_logprobs_enabled, logprobs, total_score = (
        global_config.max_retries_nl_to_sql,
        global_config.enable_logprobs,
        {},
        None,
    )

    if is_logprobs_enabled:
        output_with_logprobs = await ainvoke_logprobs(sql_chain, request_dict, config=config)  # type: ignore
        out, logprobs = output_with_logprobs.output, output_with_logprobs.logprobs
        if len(logprobs) > 0:
            total_score = compute_total_probability(logprobs["token_logprobs"])

    else:
        out = await sql_chain.ainvoke(request_dict, config=config)  # type: ignore

    if out["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.",
            failed_sql=out["query"],
        )

    return QueryResponse(
        sql=out["query"],
        sql_complexity=out["sql_complexity"],
        feedback_id="",
        logprobs=logprobs,
        total_score=total_score,
        snippet_ids=out.get("snippet_ids", []),
    )


@with_db
@with_feedback_id
async def handle_lcel_auto_query_request(
    request_dict: dict, config: dict | None = None
) -> AutoQueryResponse | NoReturn:
    """
    Async LCEL handler for /auto/query request.
    """
    from heavyiq.lcel.chains import auto_sql_chain

    max_retries = get_config().max_retries_nl_to_sql

    out = await auto_sql_chain.ainvoke(request_dict, config=config)  # type: ignore

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
    )


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

    max_retries = get_config().max_retries_nl_to_sql

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
        snippet_ids=result["snippet_ids"],
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

    max_retries = get_config().max_retries_nl_to_sql

    result = await auto_answer_chain.ainvoke(request_dict, config=config)  # type: ignore
    if not result["tables"]:
        raise NLtoTableException("No relevant tables found to answer question.")

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
        tables=result["tables"],
        snippet_ids=result["snippet_ids"],
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
    return TablesResponse(tables=result["tables"], snippet_ids=result["snippet_ids"])


@with_db
@with_feedback_id
async def handle_lcel_tables_to_questions_request(
    request_dict: dict, config: dict | None = None
) -> TablesToQuestionsResponse:
    """
    Async LCEL handler for /tables-to-questions request (ie, Tables to NL questions).

    Args:
        request_dict: Input dict
        config: Runnable config dict. Defaults to None.

    Returns:
        TablesToQuestionsResponse
    """
    from heavyiq.lcel.chains import question_chain

    config = config or {}
    configurable = config.pop("configurable", {})
    configurable.update(
        {
            "llm_temperature": request_dict["temperature"],
            "llm_n": request_dict["n"],
            "llm_max_tokens": request_dict["max_tokens"],
        }
    )

    config = {**config, "configurable": configurable}
    questions = await question_chain.ainvoke(request_dict, config=config)  # type: ignore
    return TablesToQuestionsResponse(questions=questions)


@with_db
@with_feedback_id
async def handle_lcel_cot_query_request(request_dict: dict, config: dict | None = None) -> COTQueryResponse | NoReturn:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.lcel.chains import sql_cot_chain

    global_config = get_config()

    max_retries, is_logprobs_enabled, logprobs, total_score = (
        global_config.max_retries_nl_to_sql,
        global_config.enable_logprobs,
        {},
        None,
    )

    if is_logprobs_enabled:
        output_with_logprobs = await ainvoke_logprobs(sql_cot_chain, request_dict, config=config)  # type: ignore
        out, logprobs = output_with_logprobs.output, output_with_logprobs.logprobs
        if len(logprobs) > 0:
            total_score = compute_total_probability(logprobs["token_logprobs"])

    else:
        out = await sql_cot_chain.ainvoke(request_dict, config=config)  # type: ignore

    if out["error"]:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.",
            failed_sql=out["query"],
        )

    return COTQueryResponse(
        sql=out["query"],
        cot=out["cot"],
        sql_complexity=out["sql_complexity"],
        feedback_id="",
        logprobs=logprobs,
        total_score=total_score,
    )


# Define the feedback submission handler
async def handle_submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Handle feedback submission.
    """
    from heavyiq.langchain.utils import is_langsmith_active

    if not is_langsmith_active:
        raise HTTPException(status_code=500, detail="Langsmith is not enabled in the config.")
    try:
        client = Client()
        client.create_feedback(
            request.feedback_id,
            key="feedback-key",
            score=request.score,
            comment=request.comment,
        )
        return FeedbackResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
