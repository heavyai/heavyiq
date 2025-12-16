from typing import Any

from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig
from langchain_core.runnables.configurable import RunnableConfigurableAlternatives, RunnableConfigurableFields


def merge_dicts(d1: dict, d2: dict) -> dict:
    """
    Recursively merge two dictionaries.
    """
    merged = d1.copy()
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_value_from_runnable_binding(
    binding: RunnableBinding | RunnableConfigurableAlternatives, config: RunnableConfig | None = None
) -> Any:
    """
    Gets the actual value of a runnable binded using `with_config` method.
    It's mainly used to get the underlying llm or prompt from an runnable.
    Passed config would be used to prepare the default runnable of RunnableConfigurableFields instance.
    """
    if isinstance(binding, RunnableConfigurableAlternatives):
        bind_config = binding.config
        if not bind_config:
            return binding.default
        value = binding._prepare(binding.config)
    else:
        value = binding.bound._prepare(binding.config)  # type: ignore
    if value and isinstance(value, tuple):
        actual_value, attached_config = value
        if isinstance(actual_value, RunnableConfigurableFields):
            config = config or {}
            # merge the default config and the passed config
            merged_config = merge_dicts(attached_config, config)
            return actual_value._prepare(merged_config)[0]  # type: ignore
        return actual_value
    return value


def configure_step(runnable: Runnable, run_name: str, step: str) -> Runnable:
    """
    Configure a runnable to act as an intermediate step.
    This function adds run_name, metadata to the runnable.

    Note: We use config_factories to merge with any existing configurable settings
    rather than replacing them (which would break LLM alternatives).

    Args:
        runnable: Runnable Instance
        run_name: run name
        step: step description
    """
    # Get existing config from the runnable if it has one
    existing_config = getattr(runnable, 'config', {}) or {}
    existing_configurable = existing_config.get('configurable', {})
    
    # Merge the new config with existing configurable settings
    new_config = {
        "run_name": run_name,
        "tags": ["intermediate-step"],
        "metadata": {"step": step},
        "configurable": existing_configurable,  # Preserve existing configurable settings
    }
    
    runnable_with_config = runnable.with_config(config=new_config)
    return runnable_with_config
