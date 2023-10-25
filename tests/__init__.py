import functools
from typing import Callable
from unittest.mock import patch


def override_config(func: Callable):
    """
    Decorator function added to sync testcase which overrides underlying get_config method
    to return a altered HeavyIQConfig instance from heavyiq_config fixture sync.
    """

    @functools.wraps(func)  # type: ignore
    def wrapper(*args, **kwargs):
        heavyiq_config = kwargs.get("heavyiq_config")
        if not heavyiq_config:
            raise ValueError("TestCase expects heavyiq_config fixture.")
        with patch("heavyiq.config.get_config", return_value=heavyiq_config):
            func(*args, **kwargs)

    return wrapper


def aoverride_config(func: Callable):
    """
    Decorator function added to async testcase which overrides underlying get_config method
    to return a altered HeavyIQConfig instance from heavyiq_config fixture async.
    """

    @functools.wraps(func)  # type: ignore
    async def wrapper(*args, **kwargs):
        heavyiq_config = kwargs.get("heavyiq_config")
        if not heavyiq_config:
            raise ValueError("TestCase expects heavyiq_config fixture.")
        with patch("heavyiq.config.get_config", return_value=heavyiq_config):
            await func(*args, **kwargs)

    return wrapper
