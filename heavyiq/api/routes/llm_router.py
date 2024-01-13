from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from heavyiq.api.models import CallLLMRequest, CallLLMResponse
from heavyiq.config import get_config
from heavyiq.langchain.llms import LLMType, get_llm_by_type

llmrouter = APIRouter()


@llmrouter.post("/call-llm", response_model=CallLLMResponse)
async def call_llm(request: CallLLMRequest) -> CallLLMResponse:
    """
    Call LLM.
    """
    config = get_config()
    llm = await run_in_threadpool(
        get_llm_by_type, LLMType.INSTRUCT, temperature=request.temperature, max_tokens=request.max_tokens
    )
    prompt = f"{config.custom_llm_api_instruct_prompt_start_token}{request.question}{config.custom_llm_api_instruct_prompt_end_token}"
    response = await llm.apredict(prompt, stop=request.stop)
    return CallLLMResponse(response=response.strip())
