from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from heavyiq.api.models import CallLLMRequest, CallLLMResponse
from heavyiq.langchain.llms import get_llm_by_type

llmrouter = APIRouter()


@llmrouter.post("/call-llm", response_model=CallLLMResponse)
async def call_llm(request: CallLLMRequest) -> CallLLMResponse:
    """
    Call LLM.
    """
    llm = await run_in_threadpool(
        get_llm_by_type, request.llm_type, temperature=request.temperature, max_tokens=request.max_tokens
    )
    response = await llm.apredict(request.prompt, stop=request.stop)
    return CallLLMResponse(response=response.strip())
