from typing import Any

from langchain.schema.runnable import RunnableBinding
from langchain.schema.runnable.configurable import RunnableConfigurableAlternatives


def get_value_from_runnable_binding(binding: RunnableBinding | RunnableConfigurableAlternatives) -> Any:
    """
    Gets the actual value of a runnable binded using `with_config` method.
    """
    if isinstance(binding, RunnableConfigurableAlternatives):
        return binding.default
    return binding.bound._prepare(binding.config)  # type: ignore
