import time
import asyncio
from typing import Callable, Any
from functools import wraps


def coro(f: Callable) -> Callable:
    """
    Decorator for turning async cli command functions to sync in-order to support click which by default sync.
    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        out = asyncio.run(f(*args, **kwargs))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        return out

    return wrapper
