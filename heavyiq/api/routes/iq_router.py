from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from heavyiq.langchain import HeavyDB
from heavyiq.api.dependencies import (
    valid_query_db_session,
    valid_question_db_session,
    validate_db_session_for_table_metadata,
)
from heavyiq.api.models import (
    QueryRequest,
    QueryResponse,
    QuestionResponse,
    QuestionRequest,
    FeedbackRequest,
    FeedbackResponse,
    GenerateTableMetadataRequest,
    GenerateTableMetadataResponse,
)
from heavyiq.api.handlers import (
    handle_query_request_async,
    handle_question_request_async,
    handle_submit_feedback_request_async,
    handle_generate_table_metadata_async,
    streaming_test,
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


@iqrouter.post("/generate-table-metadata", response_model=GenerateTableMetadataResponse)
async def generate_table_metadata(
    values: tuple[GenerateTableMetadataRequest, HeavyDB] = Depends(validate_db_session_for_table_metadata)
) -> GenerateTableMetadataResponse:
    """
    Endpoint which is reponsible for generating table metadata.
    """
    # Don't forget FastAPI converts Response Pydantic Object to Dict then to an instance of ResponseModel then to Dict then to JSON.
    # That's why a direct dict was returned instead of pydantic model.
    return await handle_generate_table_metadata_async(*values)


class TestRequest(BaseModel):
    message: str


@iqrouter.post("/streaming-test", response_class=Response)
def streaming_test_endpoint(value: TestRequest) -> StreamingResponse:
    response = StreamingResponse(streaming_test(value.message), media_type="text/event-stream")
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response
