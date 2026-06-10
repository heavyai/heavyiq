# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Callable, TypeVar, cast
from unittest.mock import patch

# Define a generic type variable for the function inorder to preserve the method signature
F = TypeVar("F", bound=Callable)


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
        with patch("heavyiq.config.get_config", return_value=heavyiq_config):
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
        # disable refresh cache for tables
        with patch("heavyiq.config.get_config", return_value=heavyiq_config), patch(
            "heavyiq.langchain.utils.refresh_cache_for_tables", return_value=None
        ):
            await func(*args, **kwargs)

    return cast(F, wrapper)
