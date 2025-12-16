import functools
from typing import Callable, TypeVar, cast
from unittest.mock import patch

# Define a generic type variable for the function inorder to preserve the method signature
F = TypeVar("F", bound=Callable)


def _clear_llm_cache():
    """Clear the get_llm_by_type TTL cache to ensure fresh LLM instances."""
    try:
        from heavyiq.langchain.llms import get_llm_by_type
        if hasattr(get_llm_by_type, 'cache'):
            get_llm_by_type.cache.clear()
    except Exception:
        pass


def override_config(func: F) -> F:
    """
    Decorator function added to testcase which overrides underlying get_config method
    to return a altered HeavyIQConfig instance from heavyiq_config fixture.
    """

    @functools.wraps(func)  # type: ignore
    def wrapper(*args, **kwargs):
        heavyiq_config = kwargs.get("heavyiq_config")
        if not heavyiq_config:
            raise ValueError("TestCase expects heavyiq_config fixture.")
        _clear_llm_cache()
        # Patch get_config at all locations where it's imported
        with patch("heavyiq.config.get_config", return_value=heavyiq_config), \
             patch("heavyiq.langchain.llms.get_config", return_value=heavyiq_config):
            func(*args, **kwargs)

    return cast(F, wrapper)


def aoverride_config(func: F) -> F:
    """
    async equivalent of override_config decorator.
    """

    @functools.wraps(func)  # type: ignore
    async def wrapper(*args, **kwargs):
        heavyiq_config = kwargs.get("heavyiq_config")
        if not heavyiq_config:
            raise ValueError("TestCase expects heavyiq_config fixture.")
        _clear_llm_cache()
        # Patch get_config at all locations where it's imported
        # disable refresh cache for tables
        with patch("heavyiq.config.get_config", return_value=heavyiq_config), \
             patch("heavyiq.langchain.llms.get_config", return_value=heavyiq_config), \
             patch("heavyiq.langchain.utils.refresh_cache_for_tables", return_value=None):
            await func(*args, **kwargs)

    return cast(F, wrapper)
