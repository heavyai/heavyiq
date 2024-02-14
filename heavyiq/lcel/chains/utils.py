from typing import Any

from langchain.schema.runnable import Runnable, RunnableBinding
from langchain.schema.runnable.configurable import RunnableConfigurableAlternatives


def get_value_from_runnable_binding(binding: RunnableBinding | RunnableConfigurableAlternatives) -> Any:
    """
    Gets the actual value of a runnable binded using `with_config` method.
    """
    if isinstance(binding, RunnableConfigurableAlternatives):
        return binding.default
    return binding.bound._prepare(binding.config)  # type: ignore


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
