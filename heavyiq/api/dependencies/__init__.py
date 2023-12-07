from fastapi import Body
from fastapi.concurrency import run_in_threadpool

from heavyiq.api.models import (
    GenerateTableMetadataRequest,
    NLtoAnswerRequest,
    QueryRequest,
    QuestionRequest,
    SQLtoAnswerRequest,
    TablesRequest,
)
from heavyiq.langchain import HeavyDB


async def valid_nl_to_answer_db_session(request: NLtoAnswerRequest = Body(...)) -> tuple[NLtoAnswerRequest, HeavyDB]:
    """
    Common db dependency.
    """
    db = await HeavyDB.from_session_async(request.session_id, include_tables=request.tables)  # type: ignore

    return request, db


async def valid_sql_to_answer_db_session(request: SQLtoAnswerRequest = Body(...)) -> tuple[SQLtoAnswerRequest, HeavyDB]:
    """
    Common db dependency.
    """
    db = await HeavyDB.from_session_async(request.session_id)  # type: ignore

    return request, db


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


async def validate_db_session_for_table_metadata(
    request_body: GenerateTableMetadataRequest = Body(...),
) -> tuple[GenerateTableMetadataRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of request and db objects.
    """
    db = await run_in_threadpool(
        HeavyDB.from_session, request_body.session_id, include_tables=[request_body.table_name]
    )

    return request_body, db


async def valid_tables_db_session(request: TablesRequest = Body(...)) -> tuple[TablesRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    return request, await HeavyDB.from_session_async(request.session_id)
