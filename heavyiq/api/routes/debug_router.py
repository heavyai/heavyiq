from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from heavyiq.api.models.debug import DbSessionResponse, CallLLMRequest, CallLLMResponse
from heavyiq.langchain.heavydb import HeavyDB
from heavyiq.langchain.llms import get_llm_by_type

debug_router = APIRouter()


@debug_router.get("/db-session", response_model=DbSessionResponse)
async def db_session() -> DbSessionResponse:
    """
    Get HeavyDB session id.
    """
    heavydb_sessionid = HeavyDB.create_session_id()
    return DbSessionResponse(session_id=heavydb_sessionid)


@debug_router.post("/call-llm", response_model=CallLLMResponse)
async def call_llm(request: CallLLMRequest) -> CallLLMResponse:
    """
    Call LLM.
    """
    llm = await run_in_threadpool(
        get_llm_by_type, request.llm_type, temperature=request.temperature, max_tokens=request.max_tokens
    )
    response = await llm.apredict(request.prompt)
    return CallLLMResponse(response=response)
