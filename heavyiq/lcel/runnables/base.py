import json
from abc import ABCMeta, abstractmethod
from typing import Any, AsyncIterator

from langchain.schema.runnable import Runnable, RunnableConfig

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
