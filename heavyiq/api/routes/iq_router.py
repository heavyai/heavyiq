from fastapi import APIRouter, Depends
from heavyiq.langchain import HeavyDB
from heavyiq.api.dependencies import valid_query_db_session, valid_question_db_session
from heavyiq.api.models import (
    QueryRequest,
    QueryResponse,
    QuestionResponse,
    QuestionRequest,
    FeedbackRequest,
    FeedbackResponse,
)
from heavyiq.api.handlers import (
    handle_query_request_async,
    handle_question_request_async,
    handle_submit_feedback_request_async,
)

iqrouter = APIRouter()


@iqrouter.post("/query", response_model=QueryResponse)
async def query(values: tuple[QueryRequest, HeavyDB] = Depends(valid_query_db_session)) -> QueryResponse:
    """
    Request for sql query with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QueryRequest request: Request Body
    """
    return await handle_query_request_async(*values)


@iqrouter.post("/question", response_model=QuestionResponse)
async def question(values: tuple[QuestionRequest, HeavyDB] = Depends(valid_question_db_session)) -> QuestionResponse:
    """
    Request for answer and sql with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QuestionRequest request: Request Body
    """
    return await handle_question_request_async(*values)


@iqrouter.post("/submit-feedback", response_model=FeedbackResponse)
async def submit_feedback(value: FeedbackRequest) -> FeedbackResponse:
    return await handle_submit_feedback_request_async(value)
