from heavyiq.api.handlers.decorators import with_db, with_feedback_id
from heavyiq.api.models import AnswerResponse, QueryResponse, QuestionResponse, TablesResponse


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
