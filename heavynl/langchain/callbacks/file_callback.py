import re
from typing import Any, Optional, TextIO, Pattern
from langchain.input import get_colored_text, get_bolded_text
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import FileCallbackHandler as BaseFileCallbackHandler


def print_text(
    text: str, color: Optional[str] = None, end: str = "", file: Optional[TextIO] = None, bold: bool = False
) -> None:
    """
    Print text with highlighting and no end characters and also optional log raw text to file.
    """
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

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        print_text(f"\n\n> Entering new {class_name} chain...", end="\n", file=self.file, bold=True)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print_text("\n> Finished chain.", end="\n", file=self.file, bold=True)

    def on_agent_action(self, action: AgentAction, color: Optional[str] = None, **kwargs: Any) -> Any:
        """Run on agent action."""
        print_text(action.log, color=color or self.color, file=self.file)

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
            print_text(f"\n{observation_prefix}", file=self.file)
        print_text(output, color=color or self.color, file=self.file)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}", file=self.file)

    def on_text(self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any) -> None:
        """Run when agent ends."""
        # remove color codes exists in prompt after formatting text
        if "Prompt after formatting:" in text:
            print_text(self.REMOVE_COLOR_CHARS_RGX.sub("", text), color=color or self.color, end=end, file=self.file)
        else:
            print_text(text, color=color or self.color, end=end, file=self.file)

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any) -> None:
        """Run on agent end."""
        print_text(finish.log, color=color or self.color, end="\n", file=self.file)
