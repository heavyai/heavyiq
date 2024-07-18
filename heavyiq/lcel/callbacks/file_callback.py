import json
import logging
import re
from typing import Any

from langchain.callbacks.tracers.logging import LoggingCallbackHandler
from langchain.callbacks.tracers.stdout import FunctionCallbackHandler
from langchain_core.tracers.schemas import Run
from langchain_core.tracers.stdout import elapsed
from langchain_core.utils.input import get_bolded_text, get_colored_text
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from heavyiq.logging_utils import get_heavyiq_logger


class LogFileWOColorCodeCallbackHandler(FunctionCallbackHandler):
    """Tracer that writes to the default log file."""

    name: str = "logfile_callback_handler"
    error_code: str = "\x1b[31;1m"
    remove_color_codes_rgx: re.Pattern[str] = re.compile(r"\x1b\[\d+(?:;\d+)?m")

    def __init__(self, **kwargs: Any) -> None:
        self._logger = get_heavyiq_logger()
        super().__init__(function=self.callback, **kwargs)

    @staticmethod
    def _is_error_message(text: str) -> bool:
        """
        Check whether the message received is of type error or not.
        """
        return LogFileWOColorCodeCallbackHandler.error_code in text

    def callback(self, text: str):
        """
        Callback function which was supposed to be called by the text to be printed.
        """
        # check the color code exists on the input string
        # and based on that trigger debug or error messages
        if self.__class__._is_error_message(text):
            func = self._logger.error
        else:
            func = self._logger.debug

        # remove color code before logging
        text = self.remove_color_codes_rgx.sub("", text)

        func(text, depth=1)


class CustomEncoder(json.JSONEncoder):
    """
    Custom json encoder class to deal with custom datatypes.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, VectorStoreIndex):
            return obj.index_id
        if isinstance(obj, NodeWithScore):
            return {"node_id": obj.node_id, "text": obj.get_text(), "score": obj.get_score()}
        if isinstance(obj, dict) and "repr" in obj:
            return obj["repr"]
        return super().default(obj)


def serialize_custom_dtypes(data: Any) -> Any:
    """
    Recursive func used to serialize custom dtype like VectorStoreIndex, NodeWithScore, etc.
    """
    if isinstance(data, VectorStoreIndex):
        return data.index_id
    if isinstance(data, NodeWithScore):
        return {"node_id": data.node_id, "text": data.get_text(), "score": data.get_score()}
    elif isinstance(data, list):
        return [serialize_custom_dtypes(item) for item in data]
    elif isinstance(data, dict):
        if "repr" in data:
            return data["repr"]
        return {key: serialize_custom_dtypes(value) for key, value in data.items()}
    else:
        return data


def try_json_stringify(obj: Any, fallback: str) -> str:
    """
    Try to stringify an object to JSON.
    Args:
        obj: Object to stringify.
        fallback: Fallback string to return if the object cannot be stringified.

    Returns:
        A JSON string if the object can be stringified, otherwise the fallback string.

    """
    try:
        return json.dumps(serialize_custom_dtypes(obj), indent=2, cls=CustomEncoder, ensure_ascii=False)
    except Exception:
        return fallback


class OverridedLoggingCallbackHandler(LoggingCallbackHandler):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.DEBUG,
        extra: dict | None = None,
        use_valid_runs: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        By default it logs only on DEBUG level.
        """
        # use iq logger by default
        if logger is None:
            logger = get_heavyiq_logger()  # type: ignore
        self._use_valid_runs = use_valid_runs
        super().__init__(logger, log_level, extra, **kwargs)

    def has_valid_run_name(self, run: Run) -> bool:
        """
        Check whether the run has a valid user define name.
        """
        if run.name.startswith("Runnable"):
            return False
        return True

    def _on_chain_start(self, run: Run) -> None:
        """
        Don't log on chain start.
        """
        pass

    def _on_chain_end(self, run: Run) -> None:
        """
        Log the chain end only when the elapsed time takes more than or equal to a second.
        """
        if self._use_valid_runs and not self.has_valid_run_name(run):
            return None
        elapsed_run = elapsed(run)
        json_stringify = try_json_stringify(run.outputs, "[outputs]")
        if json_stringify == "[outputs]":
            return None
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/end]', color='blue')} "
            + get_bolded_text(f"[{crumbs}] [{elapsed_run}] Exiting {run_type} run with output:\n")
            + f"{json_stringify}"
        )


LogFileCallbackHandler = OverridedLoggingCallbackHandler
