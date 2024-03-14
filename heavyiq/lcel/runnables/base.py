import json
from abc import ABCMeta, abstractmethod
from typing import Any, AsyncIterator

from langchain.schema.runnable import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig, ensure_config

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


class LLMGenerationRunnable(Runnable):
    """
    Mainly used to wrap the LanguageModel runnable to return LLMResult Generations.
    """

    def __init__(self, target: RunnableSerializable) -> None:
        super().__init__()
        self.target = target

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        return super().invoke(input, config)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[str]:
        """
        Always call the parent runnable with "ainvoke" in order to trigger this method.
        """
        config = ensure_config(config)
        # prepare and get value from the target runnable
        # this helps to create new LLM instance using the passed config options
        llm: BaseLanguageModel = get_value_from_runnable_binding(self.target, config=config)  # type: ignore
        llm_result = await llm.agenerate_prompt(
            [llm._convert_input(input)],
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            **kwargs,
        )
        generations = []
        for gen in llm_result.generations[0]:
            generations.append(gen.text)

        return generations
