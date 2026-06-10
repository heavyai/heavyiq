# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any

from langchain_classic.callbacks.tracers.logging import LoggingCallbackHandler
from langchain_classic.callbacks.tracers.stdout import FunctionCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
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


class OverridedLoggingCallbackHandler(LoggingCallbackHandler):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.DEBUG,
        extra: dict | None = None,
        **kwargs: Any
    ) -> None:
        """
        By default it logs only on DEBUG level.
        """
        # use iq logger by default
        if logger is None:
            logger = get_heavyiq_logger()  # type: ignore
        super().__init__(logger, log_level, extra, **kwargs)


LogFileCallbackHandler = OverridedLoggingCallbackHandler
