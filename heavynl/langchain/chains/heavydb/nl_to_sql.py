from __future__ import annotations
from typing import Any, Optional, List, Dict
import re

import sqlparse
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.base_language import BaseLanguageModel
from langchain.schema import HumanMessage, BaseMessage, AIMessage
from pydantic import BaseModel, Extra, Field

from heavynl.langchain import HeavyDB
from heavynl.langchain.utils import get_token_limit
from heavynl.langchain.exceptions import NLtoSQLException
from heavynl.langchain.prompts import LoggedPromptTemplate
from ..logged_llm import LoggedLLMChain

NL_TO_SQL_TEMPLATE = """Create a syntactically correct {dialect} query to answer the input question.
Only query relevant columns, avoiding SELECT * for any table.
Only create one query.
These functions DO NOT EXIST: STRING_AGG, GROUP_CONCAT
Use only existing column names from the schema description, ensuring they are from the correct table.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Do not use double colon cast operators ("::"); instead use the CAST function if needed.
Explain your thinking step-by-step in a block comment before the query. Provide your response using the format below:
Question: [QUESTION]
SQLQuery: /* step-by-step reasoning */ [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.
{table_info}
Question: {input}"""
NL_TO_SQL_PROMPT = LoggedPromptTemplate(
    name="nl_to_sql_chain",
    tags=["chain", "nl_to_sql_chain"],
    input_variables=["input", "table_info", "dialect"],
    template=NL_TO_SQL_TEMPLATE,
    version=1,
)

NL_TO_SQL_CHAT_TEMPLATE = """Create a syntactically correct {dialect} query to answer the input question.
Only query relevant columns, avoiding SELECT * for any table.
Only create one query.
These functions DO NOT EXIST: STRING_AGG, GROUP_CONCAT
Use only existing column names from the schema description, ensuring they are from the correct table.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Do not use double colon cast operators ("::"); instead use the CAST function if needed.
Explain your thinking step-by-step in a block comment before the query. Provide your response using the format below:
Question: [QUESTION]
SQLQuery: /* step-by-step reasoning */ [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.

{table_info}
"""
NL_TO_SQL_CHAT_ERROR_TEMPLATE = """
DO NOT USE "STRING_AGG", "GROUP_CONCAT" functions.
Do not use reserved SQL keywords as aliases.
Following exception appears after running your previous SQL query.

{exception}
"""
NL_TO_SQL_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(NL_TO_SQL_CHAT_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

NL_TO_SQL_ERROR_TEMPLATE = """Correct the given {dialect} query:
If a function signature does not exist, do not use it. Reformulate the query to not use that function signature.
DO NOT USE STRING_AGG, GROUP_CONCAT functions.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Do not use double colon cast operators ("::"); instead use the CAST function if needed.
Use the following format:
Question: [QUESTION]
SQLQuery: /* step-by-step thought process */ [SQL QUERY]
Error: [ERROR MESSAGE]
NewSQLQuery: /* new step-by-step thought process */ [FIXED SQL QUERY]
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
Errors: {error}
"""
NL_TO_SQL_ERROR_PROMPT = LoggedPromptTemplate(
    name="nl_to_sql_chain_error",
    tags=["chain", "nl_to_sql_chain", "nl_to_sql_chain_error"],
    input_variables=["input", "table_info", "dialect", "sql_cmd", "error"],
    template=NL_TO_SQL_ERROR_TEMPLATE,
    version=1,
)


class NLtoSQLChain(Chain, BaseModel):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    llm: BaseLanguageModel
    """LLM wrapper to use."""
    database: HeavyDB = Field(exclude=True)
    """HeavyDB Database to connect to."""
    prompt: LoggedPromptTemplate = NL_TO_SQL_PROMPT
    """Prompt to use to translate natural language to SQL."""
    error_prompt: LoggedPromptTemplate = NL_TO_SQL_ERROR_PROMPT
    """Prompt to use to fix SQL errors."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "sql"  #: :meta private:
    max_retries: int = 3

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chain"

    @property
    def input_keys(self) -> list[str]:
        """Return the input key(s).
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> dict[str, Any]:
        llm_chain = LoggedLLMChain(llm=self.llm, prompt=self.prompt)
        error_recovery_chain = LoggedLLMChain(llm=self.llm, prompt=self.error_prompt)
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        if run_manager:
            run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")
        table_info = self.database.get_table_info(table_names=table_names_to_use)

        token_limit = get_token_limit(self.llm.model_name)  # type: ignore
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(table_names=table_names_to_use, include_top_k=False)
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(table_names=table_names_to_use, include_samples=False)
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(
                table_names=table_names_to_use, include_samples=False, include_top_k=False
            )
        llm_inputs = {
            "input": input_text,
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        sql_cmd = llm_chain.predict(**llm_inputs)
        verified = False
        retries = 0
        while not verified and retries < self.max_retries:
            try:
                if run_manager:
                    run_manager.on_text(f"Verifying SQL Query: {sql_cmd}", color="blue", verbose=self.verbose)
                self.database.validate_query(sql_cmd)
                verified = True
            except Exception as e:
                if run_manager:
                    run_manager.on_text(f"Invalid SQL Query: {e}.", color="red", verbose=self.verbose)
                retry_llm_inputs = {
                    "input": input_text,
                    "sql_cmd": sql_cmd,
                    "error": f"{str(e)} \nNewSQLQuery:",
                    "dialect": self.database.dialect,
                    "table_info": table_info,
                    "stop": ["\nNewSQLQuery:"],
                }
                sql_cmd = error_recovery_chain.predict(**retry_llm_inputs)
                if sql_cmd.startswith("NewSQLQuery:"):
                    sql_cmd = sql_cmd.replace("NewSQLQuery:", "").strip()
                retries += 1
        if not verified:
            if run_manager:
                run_manager.on_text(
                    f"Failed to verify SQL query after {self.max_retries} retries.", color="red", verbose=self.verbose
                )
            raise Exception(f"Failed to verify SQL query after {self.max_retries} retries.")
        if run_manager:
            run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)

        chain_result: dict[str, Any] = {self.output_key: sql_cmd.strip()}
        return chain_result


class NLtoSQLChatChain(Chain):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    This Chain corrects invalid SQL by maintaining a history of it's previous attempts
    to create a valid SQL within the max_retries limit.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    llm: ChatOpenAI
    """LLM wrapper to use."""
    database: HeavyDB = Field(exclude=True)
    """HeavyDB Database to connect to."""
    prompt: LoggedPromptTemplate = NL_TO_SQL_CHAT_PROMPT
    """Prompt to use to translate natural language to SQL."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "sql"  #: :meta private:
    max_retries: int = 3

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chat_chain"

    @property
    def input_keys(self) -> list[str]:
        """Return the input key(s).
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys.
        :meta private:
        """
        return [self.output_key]

    def _get_table_info_wrt_token_limit(self, input_text: str, table_names_to_use: List[str]) -> str:
        """
        Gets the info of all the tables with respect to the token limit.

        Args:
            table_names_to_use (List[str]): Get table info for the list of table names.
        """
        # history = self.memory.load_memory_variables({})[self.memory.memory_key]
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        token_limit = 3800
        if isinstance(self.llm, ChatOpenAI) and self.llm.model_name == "gpt-4":
            token_limit = 7800
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(table_names=table_names_to_use, include_top_k=False)
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(table_names=table_names_to_use, include_samples=False)
        if (
            self.llm.get_num_tokens(
                self.prompt.format(input=input_text, dialect=self.database.dialect, table_info=table_info)
            )
            > token_limit
        ):
            table_info = self.database.get_table_info(
                table_names=table_names_to_use, include_samples=False, include_top_k=False
            )

        return table_info

    def _get_sql_query_from_llm_message(self, message: BaseMessage) -> str:
        """
        Helps to parse out only the SQL query from the chat_llm response.
        """
        content = message.content
        content = re.sub(r"(?is).*?SQL\s*Query:\s*|/\*.*?\*/", r"", content).split("\n\n")[0]
        return sqlparse.format(content, strip_comments=True).strip()

    def _write_message(
        self, message: str, run_manager: CallbackManagerForChainRun | None = None, color: str | None = None
    ):
        """
        Writes back the messages to terminal console.
        """
        if run_manager:
            run_manager.on_text(message, color=color, verbose=self.verbose)

    def _call(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> Dict[str, str]:
        table_names_to_use = inputs.get("tables")
        input_text = inputs[self.input_key]
        table_info = self._get_table_info_wrt_token_limit(input_text=input_text, table_names_to_use=table_names_to_use)

        messages_prompt = self.prompt.format_prompt(
            input=input_text, dialect=self.database.dialect, table_info=table_info
        )
        messages = messages_prompt.to_messages()
        self._write_message(f"Question: {input_text}\n", run_manager=run_manager, color="yellow")
        sql_message: BaseMessage = self.llm(messages)

        self._write_message("\n" + sql_message.content, run_manager=run_manager, color="blue")

        sql_valid = False
        sql_query = ""
        retries = 1

        while not sql_valid and retries <= self.max_retries:
            sql_query = self._get_sql_query_from_llm_message(sql_message)
            messages.append(AIMessage(content=f"SQL Query: {sql_query}"))
            try:
                self.database.validate_query(sql_query)
                sql_valid = True
            except Exception as exc:
                messages.append(HumanMessage(content=NL_TO_SQL_CHAT_ERROR_TEMPLATE.format(exception=exc)))
                self._write_message(
                    f"\nInvalid SQL,\n{sql_query} \n\nException:\n{exc}\nAttempt {retries}. Retrying ....",
                    run_manager=run_manager,
                    color="red",
                )
                sql_message = self.llm(messages)
                retries += 1

        if sql_valid:
            self._write_message("\n\n" + sql_query, run_manager=run_manager, color="green")
        else:
            self._write_message(
                f"\n\nFailed to verify SQL query after {self.max_retries} retries.\n",
                run_manager=run_manager,
                color="red",
            )
            raise NLtoSQLException(f"Failed to verify SQL query after {self.max_retries} retries.")

        chain_result: dict[str, Any] = {self.output_key: sql_query}
        return chain_result
