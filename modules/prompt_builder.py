from typing import NamedTuple

PromptPart = NamedTuple("PromptPart", [("label", str), ("content", str)])


class PromptBuilder:
    """Simple prompt builder that allows prompts to be pieced together"""

    _prompt_parts: list[PromptPart]

    def __init__(self, delimiter: str = "\n\n"):
        self._prompt_parts = []
        self.delimiter = delimiter

    def add_part(self, label: str, content: str):
        self._prompt_parts.append(PromptPart(label, content))

    def list_parts(self) -> list[str]:
        return [part.label for part in self._prompt_parts]

    def build(self) -> str:
        return self.delimiter.join([part.content for part in self._prompt_parts])
