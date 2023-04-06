from typing import Any, Callable
from functools import wraps


def handle_errors(func: Callable[..., Any]) -> Callable[..., tuple[Any | dict, int]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any | dict, int]:
        try:
            return func(*args, **kwargs), 200
        except Exception as e:
            return {"error": str(e)}, 500

    return wrapper
