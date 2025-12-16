import logging
import re
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from heavyiq.logging_utils import get_heavyiq_logger


class LogFileWOColorCodeCallbackHandler(BaseCallbackHandler):
    """Tracer that writes to the default log file."""

    name: str = "logfile_callback_handler"
    error_code: str = "\x1b[31;1m"
    remove_color_codes_rgx: re.Pattern[str] = re.compile(r"\x1b\[\d+(?:;\d+)?m")

    def __init__(self, **kwargs: Any) -> None:
        self._logger = get_heavyiq_logger()
        super().__init__(**kwargs)

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

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run when text is received."""
        self.callback(text)


class OverridedLoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs to a HeavyIQ logger (loguru-based)."""

    def __init__(
        self,
        logger: Optional[Any] = None,
        log_level: int = logging.DEBUG,
        extra: Optional[dict] = None,
        **kwargs: Any
    ) -> None:
        """
        By default it logs only on DEBUG level.
        """
        super().__init__(**kwargs)
        # use iq logger by default
        if logger is None:
            logger = get_heavyiq_logger()
        self.logger = logger
        self.log_level = log_level
        self.extra = extra or {}

    def _log(self, message: str) -> None:
        """Log a message at the configured log level."""
        # Map Python logging levels to HeavyIQLogger methods
        if self.log_level <= logging.DEBUG:
            self.logger.debug(message, extra=self.extra)
        elif self.log_level <= logging.INFO:
            self.logger.info(message, extra=self.extra)
        elif self.log_level <= logging.WARNING:
            self.logger.warning(message, extra=self.extra)
        else:
            self.logger.error(message, extra=self.extra)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Log LLM start."""
        self._log(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log LLM end."""
        self._log(f"LLM ended with response: {response}")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log LLM error."""
        self.logger.error(f"LLM error: {error}", extra=self.extra)

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Log chain start."""
        self._log(f"Chain started with inputs: {inputs}")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Log chain end."""
        self._log(f"Chain ended with outputs: {outputs}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log chain error."""
        self.logger.error(f"Chain error: {error}", extra=self.extra)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run when text is received."""
        self._log(text)


LogFileCallbackHandler = OverridedLoggingCallbackHandler
