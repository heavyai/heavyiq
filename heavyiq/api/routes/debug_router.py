from fastapi import APIRouter

from heavyiq.api.models.debug import DbSessionResponse
from heavyiq.langchain.heavydb import HeavyDB

debug_router = APIRouter()


@debug_router.get("/db-session", response_model=DbSessionResponse)
async def db_session() -> DbSessionResponse:
    """
    Get HeavyDB session id.
    """
    heavydb_sessionid = HeavyDB.create_session_id()
    return DbSessionResponse(session_id=heavydb_sessionid)
