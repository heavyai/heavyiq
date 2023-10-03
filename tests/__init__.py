import functools
from unittest.mock import patch


def override_config(func):
    """
    Decorator function added to testcase which overrides underlying get_config method
    to return a altered HeavyIQConfig instance from heavyiq_config fixture.
    """

    @functools.wraps(func)  # type: ignore
    def wrapper(*args, **kwargs):
        heavyiq_config = kwargs.get("heavyiq_config")
        if not heavyiq_config:
            raise ValueError("TestCase expects heavyiq_config fixture.")
        with patch("heavyiq.config.get_config", return_value=heavyiq_config):
            func(*args, **kwargs)

    return wrapper
