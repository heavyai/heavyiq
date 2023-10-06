import asyncio
from typing import Callable, Any
from functools import wraps


def coro(f: Callable) -> Callable:
    """
    Decorator for turning async cli command functions to sync in-order to support click which by default sync.
    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper
