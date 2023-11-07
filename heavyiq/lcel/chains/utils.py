from typing import Any

from langchain.schema.runnable import RunnableBinding


def get_value_from_runnable_binding(binding: RunnableBinding) -> Any:
    """
    Gets the actual value of a runnable binded using `with_config` method.
    """
    return binding.bound._prepare(binding.config)  # type: ignore
