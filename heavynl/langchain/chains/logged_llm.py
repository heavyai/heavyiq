from typing import Any

from langchain.chains import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    CallbackManager,
    Callbacks,
)

from heavynl.langchain.prompts import LoggedPromptTemplate
from heavynl.langchain.utils import is_promptlayer_active


class LoggedLLMChain(LLMChain):
    prompt: LoggedPromptTemplate

    def apply(self, input_list: list[dict[str, Any]], callbacks: Callbacks = None) -> list[dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = CallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            {"input_list": input_list},
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        if is_promptlayer_active:
            for index, inputs in enumerate(input_list):
                request_id = response.generations[index][0].generation_info["pl_request_id"]
                self.prompt.track_request(request_id, inputs)
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(self, input_list: list[dict[str, Any]], callbacks: Callbacks = None) -> list[dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = AsyncCallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = await callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            {"input_list": input_list},
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise e
        if is_promptlayer_active:
            request_id = response.generations[0][0].generation_info["pl_request_id"]
            self.prompt.track_request(request_id, input_list[0])
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs
