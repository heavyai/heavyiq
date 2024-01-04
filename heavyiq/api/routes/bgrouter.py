# This router contain endpoints that serve as triggering points for tasks running in the background.

from fastapi import APIRouter, BackgroundTasks, Request

from heavyiq.api.handlers import handle_update_index_request
from heavyiq.api.models import UpdateIndexRequest, UpdateIndexResponse

bgrouter = APIRouter()


@bgrouter.post("/update-index", response_model=UpdateIndexResponse)
async def update_index(
    http_request: Request, request: UpdateIndexRequest, background_tasks: BackgroundTasks
) -> UpdateIndexResponse:
    """
    Request for triggering vectordb (ChromaDB) create or update index.
    If a session_id is provided, the system attempts to generate documents for all tables in the database to which the session id belongs.

    - **session_id**: [Optional] HeavyDB session id.
    \f
    :param UpdateIndexRequest request: Request Body
    """
    return await handle_update_index_request(http_request, request, background_tasks)
