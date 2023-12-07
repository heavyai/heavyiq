# contain chain endpoints implemented using Langchain Expression Language (LCEL)
from fastapi import APIRouter, Depends

from heavyiq.api.dependencies import (
    valid_nl_to_answer_db_session,
    valid_query_db_session,
    valid_question_db_session,
    valid_sql_to_answer_db_session,
    valid_tables_db_session,
)
from heavyiq.api.handlers import (
    handle_lcel_nl_to_answer_request,
    handle_lcel_query_request,
    handle_lcel_question_request,
    handle_lcel_sql_to_answer_request,
    handle_lcel_tables_request,
)
from heavyiq.api.models import (
    AnswerResponse,
    NLtoAnswerRequest,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
    SQLtoAnswerRequest,
    TablesRequest,
    TablesResponse,
)
from heavyiq.api.routes.log_route import LoggingRoute
from heavyiq.langchain import HeavyDB

lcelrouter = APIRouter(route_class=LoggingRoute)


@lcelrouter.post("/query", response_model=QueryResponse)
async def query(values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)) -> QueryResponse:
    """
    Request for sql query with all the information using LCEL:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QueryRequest request: Request Body
    """
    return await handle_lcel_query_request(*values)


@lcelrouter.post("/question", response_model=QuestionResponse)
async def question(values: tuple[QuestionRequest, HeavyDB] = Depends(valid_question_db_session)) -> QuestionResponse:
    """
    Request for answer and sql with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QuestionRequest request: Request Body
    """
    return await handle_lcel_question_request(*values)


@lcelrouter.post("/nl-to-answer", response_model=AnswerResponse)
async def nl_to_answer(
    values: tuple[NLtoAnswerRequest, HeavyDB] = Depends(valid_nl_to_answer_db_session)
) -> AnswerResponse:
    """
    Request for answer with all the information:

    - **question**: NL question.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param AnswerRequest request: Request Body
    """
    return await handle_lcel_nl_to_answer_request(*values)


@lcelrouter.post("/sql-to-answer", response_model=AnswerResponse)
async def sql_to_answer(
    values: tuple[SQLtoAnswerRequest, HeavyDB] = Depends(valid_sql_to_answer_db_session)
) -> AnswerResponse:
    """
    Request for answer with all the information:

    - **query**: HeavyDB compatible SQL query.
    - **question**: NL question.
    - **session_id**: HeavyDB session id.
    \f
    :param AnswerRequest request: Request Body
    """
    return await handle_lcel_sql_to_answer_request(*values)


@lcelrouter.post("/tables", response_model=TablesResponse)
async def tables(values: tuple[TablesRequest, HeavyDB] = Depends(valid_tables_db_session)) -> TablesResponse:
    """
    Request for tables with all the information:

    - **question**: Actual NL question asked.
    - **session_id**: HeavyDB session id.
    \f
    :param AnswerRequest request: Request Body
    """
    return await handle_lcel_tables_request(*values)
