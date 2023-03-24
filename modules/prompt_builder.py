"""
Utilities for building and manipulating prompts used in chat interactions with the AI model.
"""
from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING

import modules.heavydb as db
from modules.logging_utils import default_logger as logger

if TYPE_CHECKING:
    from heavyai import Cursor

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


INTRO_PROMPT_TEMPLATES = {
    "original": "",
    "added_context": """As a language model, your task is to translate natural language instructions into SQL queries using the tables in the database described below.
The table has the following schema:""",
}

SCHEMA_PROMPT_TEMPLATES = {
    "original": "A SQL table named {table_name} is stored in the database with the following schema:",
}

INSTRUCTION_PROMPT_TEMPLATES = {
    "original": """Translate the following instruction into a SQL query on this table.
Moderate preference to return a single row with the answer.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Only output a SQL query and nothing else.""",
    "summarized_guidelines": """Translate the following instruction into a SQL query on this table.
Return a single row with the answer if possible, exclude null values, and avoid reserved SQL keywords as aliases.
Only output a SQL query and nothing else.""",
    "given_instruction": """Given a specific instruction, create an SQL query on this table, with a moderate preference for a single row answer.
Exclude null values from the results and avoid using reserved SQL keywords as aliases.
Only output a SQL query and nothing else.""",
    "with_examples": """Translate natural language instructions into SQL queries on this table (e.g., "Find the population of a country given its name.").
Return a single row with the answer if possible, exclude null values, and avoid reserved SQL keywords as aliases.
Only output a SQL query and nothing else.""",
}


def build_chatgpt_system_ask_prompt(
    table_names: list[str], include_topN: bool = True, template: str = "original"
) -> str:
    """
    Constructs a system prompt for ChatGPT to generate an SQL query based on the given table names and template.

    Args:
        table_names (list[str]): A list of table names to include in the prompt.
        include_topN (bool, optional): Include the topN for string columns with the table schema. Defaults to True.
        template (str, optional): The template key for the type of prompt to use. Defaults to "original".

    Returns:
        str: The constructed system prompt for ChatGPT.
    """
    system_prompt = PromptBuilder()
    system_prompt.add_part("intro", INTRO_PROMPT_TEMPLATES.get(template, INTRO_PROMPT_TEMPLATES.get("original") or ""))
    for table_name in table_names:
        system_prompt.add_part(
            f"{table_name}-schema-intro",
            SCHEMA_PROMPT_TEMPLATES.get(template, SCHEMA_PROMPT_TEMPLATES.get("original") or "").format(
                table_name=table_name
            ),
        )
        system_prompt.add_part(f"{table_name}-schema", db.get_table_schema(table_name))
        if include_topN:
            top_k_dict = db.get_top_k_vals_for_str_cols(table_name)
            system_prompt.add_part(
                f"{table_name}-topN",
                "\n".join([f"Example {col} values: {','.join(vals)}" for col, vals in top_k_dict.items()]),
            )
    system_prompt.add_part(
        "instruction", INSTRUCTION_PROMPT_TEMPLATES.get(template, INSTRUCTION_PROMPT_TEMPLATES.get("original") or "")
    )
    return system_prompt.build()


def build_chatgpt_system_answer_prompt(sql_result: Cursor) -> str:
    """
    Constructs a user prompt for ChatGPT to provide a plain English answer using the given SQL result.

    Args:
        sql_result (Cursor): A database cursor containing the result of the executed SQL query.

    Returns:
        str: The constructed user prompt for ChatGPT to generate an answer based on the SQL result.
    """
    column_names = [d[0] for d in sql_result.description]
    prompt_builder = PromptBuilder(delimiter="\n")
    results = sql_result.fetchall()
    prompt_builder.add_part("header", ",".join(column_names))
    for row_idx, row in enumerate(results):
        prompt_builder.add_part(f"result-table-row-{row_idx}", ",".join([str(col) for col in row]))

    logger.info("==== SQL Results ====")
    for _, part in prompt_builder._prompt_parts:
        logger.info(part)
    return prompt_builder.build()
