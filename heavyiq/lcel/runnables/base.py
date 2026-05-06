import json
from abc import ABCMeta, abstractmethod
from typing import Any, AsyncIterator, TypedDict

from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_community.llms.openai import BaseOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLMResult
from langchain_core.outputs.generation import Generation
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.runnables.utils import Input

from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.types import StepDict


class BaseRunnableWrapper(metaclass=ABCMeta):
    """
    Custom Base Runnable initialized with another runnable usually used to perform some custom actions on the data
    returned from runnable.invoke, runnable.stream_log methods.
    """

    def __init__(self, chain: Runnable):
        self.chain = chain

    @abstractmethod
    async def start(self, *args, **kwargs):
        """
        Subclasses should override this method where the necessary action should takesplace.
        """
        pass


class StreamStepsRunnableWrapper(BaseRunnableWrapper):
    """
    Runnable wrapper which helps to stream the steps involved.
    """

    async def start(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> AsyncIterator[StepDict]:
        """
        Helps to stream intermediate steps.
        """
        async for step in self.chain.astream_log(input, config=config, **kwargs, include_tags=["intermediate-step"]):
            # yield intermediate-steps and final response
            op = step.ops[0]["op"]
            if op == "add":
                step_path = step.ops[0]["path"]
                step_value = step.ops[0]["value"]
                if "tags" in step_value and "intermediate-step" in step_value["tags"]:
                    tags = step_value["tags"]
                    if any("seq:step:" in i for i in tags):
                        yield {"event": "step", "data": step_value["metadata"]["step"]}
                elif step_path == "/final_output" or step_path.startswith("/streamed_output/"):
                    yield {"event": "output", "data": json.dumps(step_value)}


class StreamLLMRunnableWrapper(BaseRunnableWrapper):
    """
    Runnable wrapper which helps to stream the llm output.
    """

    async def start(self, input: Any, config: RunnableConfig | None = None, **kwargs) -> AsyncIterator[StepDict]:
        """
        Helps to stream intermediate steps.
        """
        async for step in self.chain.astream_log(input, config=config, **kwargs, include_tags=["stream_llm"]):
            # yield intermediate-steps and final response
            op = step.ops[0]["op"]
            if op == "add":
                step_path = step.ops[0]["path"]
                step_value = step.ops[0]["value"]
                if "streamed_output/-" in step_path:
                    yield {"event": "stream", "data": step_value}
                elif step_path == "/final_output" or step_path.startswith("/streamed_output/"):
                    yield {"event": "output", "data": json.dumps(step_value)}


class LLMGeneratorItem(TypedDict):
    text: str
    logprobs: dict | None


class LLMGeneratorRunnable(Runnable):
    """
    Mainly used to wrap the LanguageModel runnable to return LLMResult Generations.
    """

    def __init__(
        self,
        llm_runnable: Any,
        name: str | None = None,
    ) -> None:
        self.llm_runnable = None
        self.llm = None
        if isinstance(llm_runnable, BaseOpenAI):
            self.llm = llm_runnable
        else:
            self.llm_runnable = llm_runnable
        self.name = name or "LLMGeneratorRunnable"

    async def _ainvoke(self, input_: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[Generation]:
        """
        Custom ainvoke method.
        """
        print(f"Invoking {self.name}")
        config = ensure_config(config)
        # prepare and get value from the target runnable
        # this helps to create new LLM instance using the passed config options
        llm: BaseLanguageModel = self.llm
        if not llm:
            llm = get_value_from_runnable_binding(self.llm_runnable, config=config)
        result: LLMResult = await llm.agenerate_prompt(
            [llm._convert_input(input_)],
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            **kwargs,
        )
        generations = []
        for gen in result.generations[0]:
            generations.append(gen)
        return generations

    def _invoke(self, input_: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[Generation]:
        """
        sync method
        """
        config = ensure_config(config)
        llm: BaseLanguageModel = self.llm
        if not llm:
            llm = get_value_from_runnable_binding(self.llm_runnable, config=config)
        result: LLMResult = llm.generate_prompt(
            [llm._convert_input(input_)],
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            **kwargs,
        )
        generations = []
        for gen in result.generations[0]:
            generations.append(gen)
        return generations

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[LLMGeneratorItem]:
        items = self._invoke(input, config=config, **kwargs)
        return [{"text": i.text, "logprobs": i.generation_info["logprobs"]} for i in items]

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[LLMGeneratorItem]:
        """
        Always call the parent runnable with "ainvoke" in order to trigger this method.
        """
        items = await self._ainvoke(input, config=config, **kwargs)
        return [{"text": i.text, "logprobs": i.generation_info["logprobs"]} for i in items]


class SessionRunnable(Runnable):
    """
    Runnable which expects session_id or any session_key should exists in the input dict.
    """

    def __init__(self, runnable: Runnable, session_key: str = "session_id"):
        self.runnable = runnable
        self.session_key = session_key

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        return super().invoke(input, config)

    async def ainvoke(self, input: Input, config: RunnableConfig | None = None) -> Any:
        session_id = input.get(self.session_key)
        assert session_id, "session_id expected"
        async with heavydb_context(session_id):
            output = await self.runnable.ainvoke(input, config)
            return output
