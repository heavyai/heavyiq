from fastapi import APIRouter
from heavynl.fastapi.models import QueryRequest, QueryResponse, QuestionResponse, QuestionRequest
from heavynl.fastapi.handlers import handle_query_request, handle_question_request, handle_query_request_async

iqrouter = APIRouter()


@iqrouter.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Request for sql query with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QueryRequest request: Request Body
    """
    return await handle_query_request_async(request)


@iqrouter.post("/question", response_model=QuestionResponse)
async def question(request: QuestionRequest) -> QuestionResponse:
    """
    Request for answer and sql with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QuestionRequest request: Request Body
    """
    return handle_question_request(request)
