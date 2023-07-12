from fastapi import APIRouter
from heavyiq import CONFIG
from heavyiq.api.models import QueryRequest, QueryResponse
from heavyiq.api.handlers import handle_query_request

iqrouter = APIRouter()


@iqrouter.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Request for sql query with all the information:

    - **question**: Actual NL question asked.
    - **tables**: List of tables to consider.
    - **session_id**: HeavyDB session id.
    \f
    :param QueryRequest request: Request Body
    """
    return handle_query_request(request)


@iqrouter.post("/question")
def question():
    # Handle POST request for endpoint2
    pass
