# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Literal

from heavyiq.config import get_config
from heavyiq.langchain.llms import LLMType
from heavyiq.lcel.prompts.default import (
    DEFAULT_INSTRUCT_PROMPT,
    DEFAULT_NL_TO_MULTIPLE_SQL_JUDGE_PROMPT,
    DEFAULT_NL_TO_SQL_COT_ERROR_PROMPT,
    DEFAULT_NL_TO_SQL_COT_PROMPT,
    DEFAULT_NL_TO_SQL_ERROR_PROMPT,
    DEFAULT_NL_TO_SQL_PROMPT,
    DEFAULT_NL_TO_TABLES_PROMPT,
    DEFAULT_SQL_TO_ANSWER_PROMPT,
    DEFAULT_TABLES_TO_QUESTIONS_PROMPT,
)


class BasePromptBuilder(metaclass=ABCMeta):
    """
    Base class which helps to build prompt template string for custom model served through vllm.
    This class is not intended for building OpenAI prompts.
    """

    _config = get_config()

    def __init__(self):
        self._prompt = None
        start_token, body, end_token = self.get_prompt_parts_by_llm_type(self.llm_type)
        default_start_token, default_body, default_end_token = self.default_prompt
        self.start_token = start_token or default_start_token
        self.body = body or default_body
        self.end_token = end_token or default_end_token

    @property
    @abstractmethod
    def llm_type(self) -> LLMType:
        """
        Subclass should define the LLM Type for which the prompt is going to built for.
        """

    @staticmethod
    def get_prompt_parts_by_llm_type(llm_type: LLMType) -> tuple[str, str, str]:
        """
        Get the prompt parts by LLM Type.
        """
        config = BasePromptBuilder._config
        if llm_type == LLMType.NL_TO_SQL:
            return (
                config.custom_prompt_nl_to_sql_start_token,
                config.custom_prompt_nl_to_sql_body,
                config.custom_prompt_nl_to_sql_end_token,
            )
        if llm_type == LLMType.NL_TO_SQL_ERROR:
            return (
                config.custom_prompt_nl_to_sql_error_start_token,
                config.custom_prompt_nl_to_sql_error_body,
                config.custom_prompt_nl_to_sql_error_end_token,
            )
        if llm_type == LLMType.SQL_TO_ANSWER:
            return (
                config.custom_prompt_sql_to_answer_start_token,
                config.custom_prompt_sql_to_answer_body,
                config.custom_prompt_sql_to_answer_end_token,
            )
        if llm_type == LLMType.NL_TO_TABLES:
            return (
                config.custom_prompt_nl_to_tables_start_token,
                config.custom_prompt_nl_to_tables_body,
                config.custom_prompt_nl_to_tables_end_token,
            )
        if llm_type == LLMType.TABLES_TO_QUESTIONS:
            return (
                config.custom_prompt_tables_to_questions_start_token,
                config.custom_prompt_tables_to_questions_body,
                config.custom_prompt_tables_to_questions_end_token,
            )
        if llm_type == LLMType.INSTRUCT:
            return (
                config.custom_prompt_instruct_start_token,
                config.custom_prompt_instruct_body,
                config.custom_prompt_instruct_end_token,
            )
        if llm_type == LLMType.NL_TO_MULTIPLE_SQL_JUDGE:
            return (
                config.custom_prompt_nl_to_multiple_sql_judge_start_token,
                config.custom_prompt_nl_to_multiple_sql_judge_body,
                config.custom_prompt_nl_to_multiple_sql_judge_end_token,
            )
        if llm_type == LLMType.NL_TO_SQL_COT:
            return (None, None, None)
        if llm_type == LLMType.NL_TO_SQL_COT_ERROR:
            return (None, None, None)
        # return None for remaining all
        return None, None, None

    @property
    @abstractmethod
    def default_prompt(self) -> tuple[str, str, str]:
        """
        Default prompt.
        """

    @cached_property
    def prompt(self) -> str:
        """
        Get or build prompt if not exists.
        """
        if self._prompt:
            return self._prompt
        self._prompt = self._build_prompt()
        return self._prompt

    @cached_property
    def openai_prompt(self) -> str:
        """
        Gets the openai prompt. ie. only the prompt body
        """
        return self.default_prompt[1]

    def _build_prompt(self) -> str:
        """
        Build the prompt from the prompt parts.
        This method should be overrided, if you want to make changes on the prompt
        based on the config parameters.
        """
        return f"{self.start_token}{self.body}{self.end_token}"


class NLtoSQLPromptBuilder(BasePromptBuilder):
    """
    Builds NL to SQL prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_SQL

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_SQL_PROMPT


class NLtoSQLErrorPromptBuilder(BasePromptBuilder):
    """
    Builds NL to SQL error prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_SQL_ERROR

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_SQL_ERROR_PROMPT


class NLtoMultipleSQLJudgePromptBuider(BasePromptBuilder):
    """
    Builds NL to multiple SQL prompt for judge llm.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_MULTIPLE_SQL_JUDGE

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_MULTIPLE_SQL_JUDGE_PROMPT


class SQLtoAnswerPromptBuilder(BasePromptBuilder):
    """
    Builds SQL to Answer prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.SQL_TO_ANSWER

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_SQL_TO_ANSWER_PROMPT


class NLtoTablesPromptBuilder(BasePromptBuilder):
    """
    Builds NL to Tables prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_TABLES

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_TABLES_PROMPT


class TablestoQuestionsPromptBuilder(BasePromptBuilder):
    """
    Builds tables to questions prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.TABLES_TO_QUESTIONS

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_TABLES_TO_QUESTIONS_PROMPT


class InstructPromptBuilder(BasePromptBuilder):
    """
    Builds instruct prompt template.
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.INSTRUCT

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_INSTRUCT_PROMPT


class NLtoSQLwithCOTPromptBuilder(BasePromptBuilder):
    """
    Builds NL to SQL with COT prompt template
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_SQL_COT

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_SQL_COT_PROMPT


class NLtoSQLwithCOTErrorPromptBuilder(BasePromptBuilder):
    """
    Builds NL to SQL with COT prompt template
    """

    @property
    def llm_type(self) -> LLMType:
        return LLMType.NL_TO_SQL_COT_ERROR

    @property
    def default_prompt(self) -> tuple[str, str, str]:
        return DEFAULT_NL_TO_SQL_COT_ERROR_PROMPT


LLMTypeBuilderMapping: dict[LLMType, BasePromptBuilder] = {
    LLMType.NL_TO_SQL: NLtoSQLPromptBuilder(),
    LLMType.NL_TO_SQL_ERROR: NLtoSQLErrorPromptBuilder(),
    LLMType.SQL_TO_ANSWER: SQLtoAnswerPromptBuilder(),
    LLMType.NL_TO_TABLES: NLtoTablesPromptBuilder(),
    LLMType.TABLES_TO_QUESTIONS: TablestoQuestionsPromptBuilder(),
    LLMType.INSTRUCT: InstructPromptBuilder(),
    LLMType.NL_TO_SQL_COT: NLtoSQLwithCOTPromptBuilder(),
    LLMType.NL_TO_SQL_COT_ERROR: NLtoSQLwithCOTErrorPromptBuilder(),
    LLMType.NL_TO_MULTIPLE_SQL_JUDGE: NLtoMultipleSQLJudgePromptBuider(),
}


def get_prompt_by_llm_type(llm_type: LLMType, prompt_type: Literal["custom", "openai"] = "custom") -> str:
    if prompt_type == "openai":
        return LLMTypeBuilderMapping[llm_type].openai_prompt
    return LLMTypeBuilderMapping[llm_type].prompt
