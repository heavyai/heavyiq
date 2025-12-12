from threading import Thread
from typing import Any, Generator
import sys
from types import ModuleType

# Compatibility shim for langchain module reorganization
# llama-index-llms-langchain still expects old module paths
if "langchain.base_language" not in sys.modules:
    import langchain_core.language_models as lc_models
    import langchain_core.messages as lc_messages
    
    # Create fake modules with the classes the old package expects
    base_language_module = ModuleType("langchain.base_language")
    base_language_module.BaseLanguageModel = lc_models.BaseLanguageModel
    sys.modules["langchain.base_language"] = base_language_module
    
    schema_module = ModuleType("langchain.schema")
    schema_module.AIMessage = lc_messages.AIMessage
    schema_module.HumanMessage = lc_messages.HumanMessage
    schema_module.SystemMessage = lc_messages.SystemMessage
    schema_module.ChatMessage = lc_messages.ChatMessage
    schema_module.FunctionMessage = lc_messages.FunctionMessage
    sys.modules["langchain.schema"] = schema_module

from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.llm import dispatcher, stream_chat_response_to_tokens, stream_completion_response_to_tokens
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.types import TokenGen
from llama_index.llms.langchain import LangChainLLM

from heavyiq.langchain.llms import LLMType, get_llm_by_type

TokenWithLogprobsGen = Generator[tuple[str, dict], None, None]


def stream_completion_response_to_tokens_with_logprobs(
    completion_response_gen: CompletionResponseGen,
) -> TokenWithLogprobsGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in completion_response_gen:
            yield response.delta, response.raw["logprobs"]

    return gen()


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

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        streaming_response = self._llm._stream(prompt, **kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            text = ""
            for chunk in streaming_response:
                text += chunk.text
                yield CompletionResponse(
                    text=text, raw={"logprobs": chunk.generation_info["logprobs"]}, delta=chunk.text
                )

        return gen()

    @dispatcher.span
    def stream(
        self, prompt: BasePromptTemplate, logprobs: bool = True, **prompt_args: Any
    ) -> TokenGen | TokenWithLogprobsGen:
        """
        Stream predict for a given prompt.
        """
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.stream_chat(messages)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = self.stream_complete(formatted_prompt, formatted=True, logprobs=logprobs)
            if logprobs:
                stream_tokens = stream_completion_response_to_tokens_with_logprobs(stream_response)
            else:
                stream_tokens = stream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens


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
