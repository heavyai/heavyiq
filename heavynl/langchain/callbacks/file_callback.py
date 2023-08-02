import asyncio
import re
from typing import Any, Literal, Optional, Pattern, TextIO, cast

import aiofiles
from aiofiles.threadpool.text import AsyncTextIOWrapper
from langchain.callbacks import FileCallbackHandler as BaseFileCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.input import get_bolded_text, get_colored_text
from langchain.schema.agent import AgentAction, AgentFinish


class AsyncFileCallbackHandler(AsyncCallbackHandler):
    """Async Callback Handler that writes to a file."""

    def __init__(self, filename: str, mode: str = "a", color: Optional[str] = None, to_stdout: bool = True) -> None:
        """
        Initialises Async file callback handler which logs the text to file and an optional stdout.

        Args:
            filename (str): log file path
            mode (str, optional): file mode. Defaults to "a".
            color (Optional[str], optional): color of the text to be prinited on console. Defaults to None.
            to_stdout (bool, optional): log to stdout as well. Defaults to True.
        """
        self.file_path = filename
        self.file = None
        self.mode = mode
        self.color = color
        self.to_stdout = to_stdout

    async def _open_file(self):
        if not self.file:
            self.file = cast(AsyncTextIOWrapper, await aiofiles.open(self.file_path, mode=self.mode))

    async def append_to_file(self, text: str):
        await self._open_file()
        if not self.file:
            raise ValueError("File instance for logging is not available.")

        await self.file.write(text)
        await self.file.flush()

    async def print_text_async(self, text: str, end: str = "", color: Optional[str] = None, bold: bool = False):
        """
        Optional write to stdout and log the text to file.
        """
        if self.to_stdout:
            bolded_text = get_bolded_text(text) if bold else text
            text_to_print = get_colored_text(bolded_text, color) if color else bolded_text
            print(text_to_print, end=end)

        await self.append_to_file(f"{text}{end}")

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        if self.file and not self.file.closed:
            asyncio.ensure_future(self.file.close())

    async def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        await self.print_text_async(f"\n\n> Entering new {class_name} chain...", end="\n", bold=True)

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        await self.print_text_async("\n> Finished chain.", end="\n", bold=True)

    async def on_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any) -> Any:
        """Run on agent action."""
        await self.print_text_async(action.log, color=color or self.color)

    async def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            await self.print_text_async(f"\n{observation_prefix}")
        await self.print_text_async(output, color=color or self.color)
        if llm_prefix is not None:
            await self.print_text_async(f"\n{llm_prefix}")

    async def on_text(self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any) -> None:
        """Run when agent ends."""
        await self.print_text_async(text, color=color or self.color, end=end)

    async def on_agent_finish(self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        """Run on agent end."""
        await self.print_text_async(finish.log, color=color or self.color, end="\n")


def print_text(
    text: str,
    color: Optional[str] = None,
    end: str = "",
    file: Optional[TextIO] = None,
    bold: bool = False,
    to_stdout: bool = True,
) -> None:
    """
    Print text with highlighting and no end characters and also optional log raw text to file.
    """
    if to_stdout:
        bolded_text = get_bolded_text(text=text) if bold else text
        text_to_print = get_colored_text(bolded_text, color) if color else bolded_text
        print(text_to_print, end=end)
    if file:
        # don't add color chars
        file.write(f"{text}{end}")
        file.flush()  # ensure all content are written to file


class FileCallbackHandler(BaseFileCallbackHandler):
    """
    Custom file callback handler class reponsible for stdout and file logging.
    It also make sure to remove terminal color characters before wiring to a file.
    """

    REMOVE_COLOR_CHARS_RGX: Pattern[str] = re.compile(r"\x1b\[(?:\d+;)?\dm")

    def __init__(self, filename: str, mode: str = "a", color: str | None = None, to_stdout: bool = True) -> None:
        """Initialize callback handler."""
        super().__init__(filename=filename, mode=mode, color=color)
        # whether to print to stdout or not
        self.to_stdout = to_stdout

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print_text(
            f"\n\n> Entering new {class_name} chain...", end="\n", file=self.file, bold=True, to_stdout=self.to_stdout
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print_text("\n> Finished chain.", end="\n", file=self.file, bold=True, to_stdout=self.to_stdout)

    def on_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color or self.color, file=self.file, to_stdout=self.to_stdout)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}", file=self.file, to_stdout=self.to_stdout)
        print_text(output, color=color or self.color, file=self.file, to_stdout=self.to_stdout)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}", file=self.file, to_stdout=self.to_stdout)

    def on_text(self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any) -> None:
        """Run when agent ends."""
        # remove color codes exists in prompt after formatting text
        if "Prompt after formatting:" in text:
            print_text(
                self.REMOVE_COLOR_CHARS_RGX.sub("", text),
                color=color or self.color,
                end=end,
                file=self.file,
                to_stdout=self.to_stdout,
            )
        else:
            print_text(text, color=color or self.color, end=end, file=self.file, to_stdout=self.to_stdout)

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color or self.color, end="\n", file=self.file, to_stdout=self.to_stdout)
