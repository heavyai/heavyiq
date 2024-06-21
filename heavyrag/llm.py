from typing import Any

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.llms.langchain import LangChainLLM

from heavyiq.langchain.llms import LLMType, get_llm_by_type


class OverridedLangChainLLM(LangChainLLM):
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        output_str = self._llm.invoke(prompt, **kwargs)
        return CompletionResponse(text=output_str)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        output_str = await self._llm.ainvoke(prompt, **kwargs)
        return CompletionResponse(text=output_str)


def get_llm(temperature: float = 0.1, max_tokens: int = 512) -> LangChainLLM:
    """
    Gets the model for handling RAG QA prompts.
    """
    return OverridedLangChainLLM(llm=get_llm_by_type(LLMType.RAG, temperature=temperature, max_tokens=max_tokens))


def get_rerank_llm(temperature: float = 0.0, max_tokens: int = 1024, **kwargs) -> LangChainLLM:
    """
    Gets the model for handling RAG reranking of nodes.
    """
    return OverridedLangChainLLM(
        llm=get_llm_by_type(LLMType.RAG, temperature=temperature, max_tokens=max_tokens, **kwargs)
    )
