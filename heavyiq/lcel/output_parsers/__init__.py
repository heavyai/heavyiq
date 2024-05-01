import re
from json import JSONDecodeError
from typing import Any, Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers import json as jop
from langchain_core.outputs import Generation


def parse_json_markdown(json_string: str, *, parser: Callable[[str], Any] = jop.parse_partial_json) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Try to find JSON string within triple backticks
    match = re.search(r"```(json)(.*)", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip().strip("`")

    # handle newlines and other special characters inside the returned value
    json_str = jop._custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    parsed = parser(json_str)

    return parsed


class OverrideJsonOutputParser(JsonOutputParser):
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e


class COTJsonOutputParser(JsonOutputParser):
    """
    Custom JSON parser for parsing SQL COT output.
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        cot, sql = "", ""
        out = text.split("** SQL Query **")
        if len(out) == 1:
            cot = out[0]
        else:
            cot, sql = out
        cot = cot.strip()
        return {"cot": cot.split("\n") if cot else [], "sql": sql}
