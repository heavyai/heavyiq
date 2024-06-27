from typing import Any

from langchain.schema.runnable import Runnable, RunnableBinding, RunnableConfig
from langchain.schema.runnable.configurable import RunnableConfigurableAlternatives, RunnableConfigurableFields


def get_value_from_runnable_binding(
    binding: RunnableBinding | RunnableConfigurableAlternatives, config: RunnableConfig | None = None
) -> Any:
    """
    Gets the actual value of a runnable binded using `with_config` method.
    It's mainly used to get the underlying llm or prompt from an runnable.
    Passed config would be used to prepare the default runnable of RunnableConfigurableFields instance.
    """
    if isinstance(binding, RunnableConfigurableAlternatives):
        return binding.default
    value = binding.bound._prepare(binding.config)  # type: ignore
    if value and isinstance(value, tuple):
        actual_value, attached_config = value
        if isinstance(actual_value, RunnableConfigurableFields):
            return actual_value._prepare(config or attached_config)[0]  # type: ignore
        return actual_value
    return value


def configure_step(runnable: Runnable, run_name: str, step: str) -> Runnable:
    """
    Configure a runnable to act as an intermediate step.
    This function adds run_name, metadata to the runnable.

    Args:
        runnable: Runnable Instance
        run_name: run name
        step: step description
    """
    runnable_with_config = runnable.with_config(
        config={
            "run_name": run_name,
            "tags": ["intermediate-step"],
            "metadata": {"step": step},
        }
    )
    return runnable_with_config
