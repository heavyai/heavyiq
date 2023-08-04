from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from heavyiq.api.models import QueryRequest, QuestionRequest
from heavyiq.langchain import HeavyDB


async def valid_query_db_session(query_request: QueryRequest = Body(...)) -> tuple[QueryRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    db = await run_in_threadpool(HeavyDB.from_session, query_request.session_id, include_tables=query_request.tables)

    return query_request, db


async def valid_question_db_session(question_request: QuestionRequest = Body(...)) -> tuple[QuestionRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    db = await run_in_threadpool(
        HeavyDB.from_session, question_request.session_id, include_tables=question_request.tables
    )

    return question_request, db
