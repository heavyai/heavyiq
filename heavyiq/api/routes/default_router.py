import os

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse

from heavyiq.api.models import CallLLMRequest, CallLLMResponse
from heavyiq.langchain.llms import get_llm_by_type

defaultrouter = APIRouter()

PUBLIC_PATH = "public"


@defaultrouter.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@defaultrouter.get("/version.txt")
async def serve_version() -> FileResponse:
    file_path = os.path.join(PUBLIC_PATH, "version.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Not Found")

    return FileResponse(file_path)


@defaultrouter.post("/call-llm", response_model=CallLLMResponse)
async def call_llm(request: CallLLMRequest) -> CallLLMResponse:
    """
    Call LLM.
    """
    llm = await run_in_threadpool(
        get_llm_by_type, request.llm_type, temperature=request.temperature, max_tokens=request.max_tokens
    )
    response = await llm.apredict(request.prompt, stop=request.stop)
    return CallLLMResponse(response=response.strip())
