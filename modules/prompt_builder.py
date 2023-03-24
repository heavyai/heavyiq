from typing import NamedTuple

PromptPart = NamedTuple("PromptPart", [("label", str), ("content", str)])


class PromptBuilder:
    """
    A class that helps build prompts by concatenating individual parts with a specified delimiter.
    Each part has a label for easy identification and content that will be included in the final prompt.

    Attributes:
        _prompt_parts (list[PromptPart]): A list of prompt parts (label and content) to be combined.
        delimiter (str): A delimiter used to separate the content of each part in the final prompt.
    """

    def __init__(self, delimiter: str = "\n\n"):
        """
        Initializes a new instance of the PromptBuilder class.

        Args:
            delimiter (str, optional): A delimiter used to separate the content of each part in the final prompt.
                                        Defaults to two newline characters.
        """
        self._prompt_parts: list[PromptPart] = []
        self.delimiter: str = delimiter

    def add_part(self, label: str, content: str):
        """
        Adds a new part to the prompt with the given label and content.

        Args:
            label (str): The label for the new part.
            content (str): The content of the new part.
        """
        self._prompt_parts.append(PromptPart(label, content))

    def remove_part(self, label: str):
        """
        Removes a part from the prompt by its label.

        Args:
            label (str): The label of the part to be removed.
        """
        self._prompt_parts = [part for part in self._prompt_parts if part.label != label]

    def list_parts(self) -> list[str]:
        """
        Returns a list of labels for all parts currently added to the prompt.

        Returns:
            list[str]: A list of labels for all parts.
        """
        return [part.label for part in self._prompt_parts]

    def build(self) -> str:
        """
        Builds the final prompt by concatenating the content of each part with the specified delimiter.

        Returns:
            str: The final prompt as a single string.
        """
        return self.delimiter.join([part.content for part in self._prompt_parts])
