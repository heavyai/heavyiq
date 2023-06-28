from functools import wraps
from typing import Any, Callable
from heavynl.logging_utils import heavynl_logger as logger


def handle_errors(func: Callable[..., Any]) -> Callable[..., tuple[Any | dict, int]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any | dict, int]:
        try:
            return func(*args, **kwargs), 200
        except Exception as e:
            logger.error(str(e))
            return {"error": str(e)}, 500

    return wrapper
