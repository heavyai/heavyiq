from fastapi import Body

from heavyiq.api.models import (
    AnswerRequest,
    AutoQueryRequest,
    AutoQuestionRequest,
    GenerateTableMetadataRequest,
    GenerateVegaLiteRequest,
    QueryRequest,
    QuestionRequest,
    TablesRequest,
    TablesToQuestionsRequest,
)
from heavyiq.langchain import HeavyDB


async def valid_vega_db_session(
    request: GenerateVegaLiteRequest = Body(...),
) -> tuple[GenerateVegaLiteRequest, HeavyDB]:
    """
    Common db dependency.
    """
    db = await HeavyDB.from_session_async(request.session_id, include_tables=request.tables)  # type: ignore
    return request, db


async def valid_answer_db_session(request: AnswerRequest = Body(...)) -> tuple[AnswerRequest, HeavyDB]:
    """
    DB dependency for answer endpoint.
    """
    db = await HeavyDB.from_session_async(request.session_id, include_tables=request.tables)  # type: ignore

    return request, db


async def valid_query_db_session(query_request: QueryRequest = Body(...)) -> tuple[QueryRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    db = await HeavyDB.from_session_async(query_request.session_id, include_tables=query_request.tables)

    return query_request, db


async def valid_question_db_session(question_request: QuestionRequest = Body(...)) -> tuple[QuestionRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    db = await HeavyDB.from_session_async(question_request.session_id, include_tables=question_request.tables)

    return question_request, db


async def validate_db_session_for_table_metadata(
    request_body: GenerateTableMetadataRequest = Body(...),
) -> tuple[GenerateTableMetadataRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of request and db objects.
    """
    db = await HeavyDB.from_session_async(request_body.session_id, include_tables=[request_body.table_name])

    return request_body, db


async def valid_tables_db_session(request: TablesRequest = Body(...)) -> tuple[TablesRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    return request, await HeavyDB.from_session_async(request.session_id)


async def valid_auto_query_db_session(request: AutoQueryRequest = Body(...)) -> tuple[AutoQueryRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of auto query (ie. auto detect tables) request and db objects.
    """
    return request, await HeavyDB.from_session_async(request.session_id)


async def valid_auto_question_db_session(
    request: AutoQuestionRequest = Body(...),
) -> tuple[AutoQuestionRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of auto question request and db objects.
    """
    return request, await HeavyDB.from_session_async(request.session_id)


async def valid_tables_to_questions_db_session(
    request: TablesToQuestionsRequest = Body(...),
) -> tuple[TablesToQuestionsRequest, HeavyDB]:
    """
    Dependency where the injected handler function gets a tuple of query request and db objects.
    """
    return request, await HeavyDB.from_session_async(request.session_id, include_tables=request.tables)
