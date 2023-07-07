from functools import wraps
from typing import Any, Callable

from heavynl.logging_utils import get_heavynl_logger


def handle_errors(func: Callable[..., Any]) -> Callable[..., tuple[Any | dict, int]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any | dict, int]:
        logger = get_heavynl_logger()
        try:
            return func(*args, **kwargs), 200
        except Exception as e:
            logger.exception(str(e))
            return {"error": str(e)}, 500

    return wrapper
