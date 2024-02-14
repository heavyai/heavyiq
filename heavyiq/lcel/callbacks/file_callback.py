import re
from typing import Any

from langchain.callbacks.tracers.stdout import FunctionCallbackHandler

from heavyiq.logging_utils import get_heavyiq_logger


class LogFileCallbackHandler(FunctionCallbackHandler):
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
        return LogFileCallbackHandler.error_code in text

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
