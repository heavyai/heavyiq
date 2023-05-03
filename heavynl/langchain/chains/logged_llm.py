from typing import Any

from langchain.chains import LLMChain

from heavynl.langchain.prompts import LoggedPromptTemplate
from heavynl.langchain.utils import is_promptlayer_active


class LoggedLLMChain(LLMChain):
    prompt: LoggedPromptTemplate

    def apply(self, input_list: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = self.generate(input_list)
        if is_promptlayer_active:
            for index, inputs in enumerate(input_list):
                request_id = response.generations[index][0].generation_info["pl_request_id"]
                self.prompt.track_request(request_id, inputs)
        return self.create_outputs(response)

    async def aapply(self, input_list: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = await self.agenerate(input_list)
        if is_promptlayer_active:
            request_id = response.generations[0][0].generation_info["pl_request_id"]
            self.prompt.track_request(request_id, input_list[0])
        return self.create_outputs(response)
