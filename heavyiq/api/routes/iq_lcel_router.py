# contain chain endpoints implemented using Langchain Expression Language (LCEL)
from fastapi import APIRouter, Depends

from heavyiq.api.dependencies import (
    valid_answer_db_session,
    valid_auto_query_db_session,
    valid_auto_question_db_session,
    valid_query_db_session,
    valid_question_db_session,
    valid_tables_db_session,
    valid_tables_to_questions_db_session,
)
from heavyiq.api.handlers import (
    handle_lcel_answer_request,
    handle_lcel_auto_query_request,
    handle_lcel_auto_question_request,
    handle_lcel_query_request,
    handle_lcel_question_request,
    handle_lcel_tables_request,
    handle_lcel_tables_to_questions_request,
)
from heavyiq.api.models import (
    AnswerRequest,
    AnswerResponse,
    AutoQueryRequest,
    AutoQueryResponse,
    AutoQuestionRequest,
    AutoQuestionResponse,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
    TablesRequest,
    TablesResponse,
    TablesToQuestionsRequest,
    TablesToQuestionsResponse,
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


@lcelrouter.post("/auto/query", response_model=AutoQueryResponse)
async def auto_query(
    values: tuple[AutoQueryRequest, HeavyDB] = Depends(valid_auto_query_db_session)
) -> AutoQueryResponse:
    """
    Request for sql query with all the information using LCEL:

    - **question**: Actual NL question asked.
    - **session_id**: HeavyDB session id.
    - **allowed_tables**: Optional List of tables to consider.
    \f
    :param AutoQueryRequest request: Request Body
    """
    return await handle_lcel_auto_query_request(*values)


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


@lcelrouter.post("/auto/question", response_model=AutoQuestionResponse)
async def auto_question(
    values: tuple[AutoQuestionRequest, HeavyDB] = Depends(valid_auto_question_db_session)
) -> AutoQuestionResponse:
    """
    Request for answer and sql with all the information:

    - **question**: Actual NL question asked.
    - **session_id**: HeavyDB session id.
    - **allowed_tables**: Optional List of tables to consider.
    \f
    :param AutoQuestionRequest request: Request Body
    """
    return await handle_lcel_auto_question_request(*values)


@lcelrouter.post("/answer", response_model=AnswerResponse)
async def answer(values: tuple[AnswerRequest, HeavyDB] = Depends(valid_answer_db_session)) -> AnswerResponse:
    """
    Request for answer with all the information:

    - **query**: HeavyDB compatible SQL query.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param AnswerRequest request: Request Body
    """
    return await handle_lcel_answer_request(*values)


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


@lcelrouter.post("/tables-to-questions", response_model=TablesToQuestionsResponse)
async def tables_to_questions(
    values: tuple[TablesToQuestionsRequest, HeavyDB] = Depends(valid_tables_to_questions_db_session)
) -> TablesToQuestionsResponse:
    """
    Request to generate questions for the given tables with all the information:

    - **tables**: Tables to consider for generating questions.
    - **session_id**: HeavyDB session id.
    \f
    :param TablesToQuestionsRequest: Request Body
    """
    return await handle_lcel_tables_to_questions_request(*values)
