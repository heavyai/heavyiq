from __future__ import annotations

from typing import Any, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from pydantic import Extra, Field

from heavynl.langchain import HeavyDB
from heavynl.langchain.utils import get_table_info_wrt_token_limit
from heavynl.langchain.chains import BaseChain
from heavynl.langchain.exceptions import NLtoSQLException
from heavynl.langchain.prompts import LoggedChatPromptTemplate, LoggedPromptTemplate
from heavynl.utils import strip_sql_comments

from ..logged_llm import LoggedLLMChain

NL_TO_SQL_TEMPLATE = """Create a syntactically correct {dialect} query to answer the input question.
Only query relevant columns, avoiding SELECT * for any table.
Only create one query.
These functions DO NOT EXIST: STRING_AGG, GROUP_CONCAT
Use only existing column names from the schema description, ensuring they are from the correct table.
Exclude null values from the results. DO NOT use reserved SQL keywords as aliases.
DO NOT use double colon cast operators ("::"); instead use the CAST function if needed.
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
Exclude null values from the results. DO NOT use reserved SQL keywords as aliases.
DO NOT use double colon cast operators ("::"); instead use the CAST function if needed.
Explain your thinking step-by-step in a block comment before the query. Provide your response using the format below:
Question: [QUESTION]
SQLQuery: /* step-by-step reasoning */ [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.

{table_info}
"""
NL_TO_SQL_CHAT_ERROR_TEMPLATE = """
DO NOT USE "STRING_AGG", "GROUP_CONCAT" functions.
DO NOT use reserved SQL keywords as aliases.
Following exception appears after running your previous SQL query.

{exception}
"""
NL_TO_SQL_CHAT_MESSAGES = [
    SystemMessagePromptTemplate.from_template(NL_TO_SQL_CHAT_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{input}"),
]

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


class BaseNltoSQLChain(BaseChain):
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

    llm: BaseLanguageModel | BaseChatModel
    """LLM wrapper to use."""
    database: HeavyDB = Field(exclude=True)
    """HeavyDB Database to connect to."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "sql"  #: :meta private:
    max_retries: int = 3

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


class NLtoSQLChain(BaseNltoSQLChain):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseLanguageModel
    prompt: LoggedPromptTemplate = NL_TO_SQL_PROMPT
    """Prompt to use to translate natural language to SQL."""
    error_prompt: LoggedPromptTemplate = NL_TO_SQL_ERROR_PROMPT
    """Prompt to use to fix SQL errors."""

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chain"

    def _call(self, inputs: dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> dict[str, Any]:
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        self.write_callback_message(input_text, run_manager=run_manager)
        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")
        create_query_prompt = self.prompt.partial(input=input_text, dialect=self.database.dialect)
        table_info = get_table_info_wrt_token_limit(self.llm, self.database, create_query_prompt, table_names_to_use)
        llm_inputs = {
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        llm_chain = LoggedLLMChain(llm=self.llm, prompt=create_query_prompt)
        sql_cmd = llm_chain.predict(**llm_inputs)
        verified = False
        retries = 0
        while not verified and retries < self.max_retries:
            try:
                self.write_callback_message(f"Verifying SQL Query: {sql_cmd}", run_manager=run_manager, color="blue")
                self.database.validate_query(sql_cmd)
                verified = True
            except Exception as e:
                self.write_callback_message(f"Invalid SQL Query: {e}.", run_manager=run_manager, color="red")
                error_prompt = self.error_prompt.partial(
                    input=input_text,
                    sql_cmd=sql_cmd,
                    error=f"{str(e)[0:150]} \nNewSQLQuery:",
                    dialect=self.database.dialect,
                )
                table_info = get_table_info_wrt_token_limit(self.llm, self.database, error_prompt, table_names_to_use)
                error_recovery_chain = LoggedLLMChain(llm=self.llm, prompt=error_prompt)
                retry_llm_inputs = {
                    "table_info": table_info,
                    "stop": ["\nNewSQLQuery:"],
                }
                sql_cmd = error_recovery_chain.predict(**retry_llm_inputs)
                if sql_cmd.startswith("NewSQLQuery:"):
                    sql_cmd = sql_cmd.replace("NewSQLQuery:", "").strip()
                retries += 1
        if not verified:
            self.write_callback_message(
                f"Failed to verify SQL query after {self.max_retries} retries.", run_manager=run_manager, color="red"
            )
            raise NLtoSQLException(f"Failed to verify SQL query after {self.max_retries} retries.")

        self.write_callback_message(sql_cmd, run_manager=run_manager, color="green")

        chain_result: dict[str, Any] = {self.output_key: sql_cmd.strip()}
        return chain_result


class NLtoSQLChatChain(BaseNltoSQLChain):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    This Chain corrects invalid SQL by maintaining a history of it's previous attempts
    to create a valid SQL within the max_retries limit.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseChatModel
    """LLM wrapper to use."""
    prompt_messages: list[BaseMessagePromptTemplate | BaseMessage] = NL_TO_SQL_CHAT_MESSAGES
    """Prompt to use to translate natural language to SQL."""
    messages_with_comments: bool = True  # helps to keep comments on the messages that we going to stack up

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chat_chain"

    def get_sql_query(self, return_message: BaseMessage) -> str:
        """
        Helps to parse out SQL query from the llm response message.
        """
        return strip_sql_comments(return_message.content.split("\nSQLQuery:")[1].strip())

    def build_ai_message(self, return_message: BaseMessage) -> AIMessage:
        """
        Helps to build AI Message from the llm returned response message.
        """
        if self.messages_with_comments:
            return AIMessage(content=return_message.content)
        # don't allow comments on messages
        return AIMessage(content=f"SQLQuery: {self.get_sql_query(return_message)}")

    def _call(self, inputs: dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> dict[str, str]:
        table_names_to_use: list[str] | None = inputs.get("tables")  # type: ignore
        input_text = inputs[self.input_key]
        messages_prompt = LoggedChatPromptTemplate.from_messages(
            messages=self.prompt_messages,
            name="nl_to_sql_chat_chain",
            tags=["chain", "nl_to_sql_chat_chain"],
            input_variables=["table_info"],
            partial_variables={"dialect": self.database.dialect, "input": input_text},
            version=1,
        )
        table_info = get_table_info_wrt_token_limit(self.llm, self.database, messages_prompt, table_names_to_use)

        messages_prompt = messages_prompt.format_prompt(table_info=table_info)
        messages = messages_prompt.to_messages()
        self.write_callback_message(f"Question: {input_text}\n", run_manager=run_manager, color="yellow")
        sql_message: BaseMessage = self.llm(messages)
        self.write_callback_message("\n" + sql_message.content, run_manager=run_manager, color="blue")

        sql_valid = False
        sql_query = ""
        retries = 1

        while not sql_valid and retries <= self.max_retries:
            sql_query = self.get_sql_query(sql_message)
            messages.append(self.build_ai_message(sql_message))
            try:
                self.database.validate_query(sql_query)
                sql_valid = True
            except Exception as exc:
                messages.append(HumanMessage(content=NL_TO_SQL_CHAT_ERROR_TEMPLATE.format(exception=str(exc)[0:150])))
                self.write_callback_message(
                    f"\nInvalid SQL,\n{sql_query} \n\nException:\n{exc}\nAttempt {retries}. Retrying ....",
                    run_manager=run_manager,
                    color="red",
                )
                sql_message = self.llm(messages)
                retries += 1

        if sql_valid:
            self.write_callback_message("\n\n" + sql_query, run_manager=run_manager, color="green")
        else:
            self.write_callback_message(
                f"\n\nFailed to verify SQL query after {self.max_retries} retries.\n",
                run_manager=run_manager,
                color="red",
            )
            raise NLtoSQLException(f"Failed to verify SQL query after {self.max_retries} retries.")

        chain_result: dict[str, Any] = {self.output_key: sql_query}
        return chain_result


def get_nl_to_sql_chain_by_llm(llm: BaseLanguageModel) -> type[BaseNltoSQLChain]:
    """
    Gets the appropriate nt_to_sql chain class based upon the llm passed.
    """
    return NLtoSQLChatChain if isinstance(llm, BaseChatModel) else NLtoSQLChain
