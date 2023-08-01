from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)
from langchain.schema import LLMResult

from heavynl.langchain.prompts import LoggedPromptTemplate
from heavynl.langchain.utils import is_promptlayer_active
from heavynl.langchain.chains import FileCallbackHandlerForChainMixin


class LoggedLLMChain(FileCallbackHandlerForChainMixin, LLMChain):
    prompt: LoggedPromptTemplate

    def generate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        res = self.llm.generate_prompt(prompts, stop, callbacks=run_manager.get_child() if run_manager else None)
        if is_promptlayer_active:
            for index, inputs in enumerate(input_list):
                generation_info = res.generations[index][0].generation_info
                if generation_info is not None:
                    request_id = generation_info["pl_request_id"]
                    self.prompt.track_request(request_id, inputs)
        return res

    async def agenerate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        res = await self.llm.agenerate_prompt(prompts, stop, callbacks=run_manager.get_child() if run_manager else None)
        if is_promptlayer_active:
            for index, inputs in enumerate(input_list):
                generation_info = res.generations[index][0].generation_info
                if generation_info is not None:
                    request_id = generation_info["pl_request_id"]
                    self.prompt.track_request(request_id, inputs)
        return res
