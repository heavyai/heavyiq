from __future__ import annotations

from typing import Any, Optional

from pydantic import Extra

import re

from fastapi.concurrency import run_in_threadpool
from langchain.chat_models.base import BaseChatModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, BaseChatPromptTemplate
from langchain.schema import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from heavyiq.config import get_config
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.heavydb import HeavyDB
from heavyiq.langchain.chains import BaseChain
from heavyiq.langchain.utils import populate_table_info_wrt_token_limit
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.utils import strip_sql_comments

NL_TO_SQL_TEMPLATE = """Create a syntactically correct SQL query to answer the input question.
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
Question: {input}
SQLQuery:"""
NL_TO_SQL_PROMPT = PromptTemplate.from_template(NL_TO_SQL_TEMPLATE)

CUSTOM_LLM_NL_TO_SQL_TEMPLATE = """<|sql prompt|>
You are a experienced data analyst adept at writing SQL queries to answer user questions.

You have access to the following relational tables, with schemas below.
{table_info}
Write a SQL query to answer the following question:
{input}
<|sql answer|>"""
CUSTOM_LLM_NL_TO_SQL_PROMPT = PromptTemplate.from_template(CUSTOM_LLM_NL_TO_SQL_TEMPLATE)

NL_TO_SQL_ERROR_TEMPLATE = """Correct the given SQL query:
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
Error: {error}
NewSQLQuery:
"""
NL_TO_SQL_ERROR_PROMPT = PromptTemplate.from_template(NL_TO_SQL_ERROR_TEMPLATE)

CUSTOM_NL_TO_SQL_ERROR_TEMPLATE = """<|sql error prompt|>
You generated a SQL query that generated an exception when executed in the HeavyDB database.
You have access to the following relation tables, with schemas below.
{table_info}
In attempting to answer the following user question:

{input},
you generated the following SQL query:
{sql_cmd}
, which failed to run in the HeavyDB database, generating the following error:
{error}

Please alter the query to run without error in HeavyDB:
<|sql error answer|>"""

CUSTOM_NL_TO_SQL_ERROR_PROMPT = PromptTemplate.from_template(CUSTOM_NL_TO_SQL_ERROR_TEMPLATE)


NL_TO_SQL_CHAT_TEMPLATE = """Create a syntactically correct SQL query to answer the input question.
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
NL_TO_SQL_CHAT_MESSAGES = [
    SystemMessagePromptTemplate.from_template(NL_TO_SQL_CHAT_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{input}"),
]
NL_TO_SQL_CHAT_PROMPT = ChatPromptTemplate.from_messages(NL_TO_SQL_CHAT_MESSAGES)

NL_TO_SQL_CHAT_ERROR_TEMPLATE = """
DO NOT USE "STRING_AGG", "GROUP_CONCAT" functions.
DO NOT use reserved SQL keywords as aliases.
Following exception appears after running your previous SQL query.

{exception}
"""


class BaseNLtoSQLChain(BaseChain):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseLanguageModel | BaseChatModel
    database: HeavyDB
    input_key: str = "query"  #: :meta private:
    output_key: str = "sql"  #: :meta private:
    output_complexity_key: str = "sql_complexity"  #: :meta private:
    max_retries: int = 3

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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


class NLtoSQLChain(BaseNLtoSQLChain):
    """
    Chain for converting a natural language question to a SQL query designed to answer the question with its results.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseLanguageModel
    prompt: BasePromptTemplate
    error_prompt: BasePromptTemplate = NL_TO_SQL_ERROR_PROMPT
    """Prompt object to use."""

    def __init__(self, *args, **kwargs):
        if is_using_custom_trained_llm():
            prompt = CUSTOM_LLM_NL_TO_SQL_PROMPT
            error_prompt = CUSTOM_NL_TO_SQL_ERROR_PROMPT
        else:
            prompt = NL_TO_SQL_PROMPT
            error_prompt = NL_TO_SQL_ERROR_PROMPT
        super().__init__(*args, prompt=prompt, error_prompt=error_prompt, **kwargs)  # type: ignore

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        self.write_callback_message(inputs[self.input_key], run_manager=run_manager)

        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")

        partial_gen_sql_prompt = self.prompt.partial(input=inputs[self.input_key])
        gen_sql_prompt = populate_table_info_wrt_token_limit(
            partial_gen_sql_prompt, self.llm, self.database, table_names_to_use
        )

        response = self.llm.generate_prompt(
            [gen_sql_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        sql_cmd = response.generations[0][0].text.strip()  # type: ignore
        verified = False
        retries = 0

        while not verified:
            try:
                self.write_callback_message(f"Verifying SQL Query: {sql_cmd}", run_manager=run_manager, color="blue")
                self.database.validate_query(sql_cmd)
                verified = True
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    break
                self.write_callback_message(f"Invalid SQL Query: {e}.", run_manager=run_manager, color="red")
                
                truncated_error = str(e)[0:150] if len(str(e)) > 150 else str(e)
                partial_error_prompt = self.error_prompt.partial(
                    input=inputs[self.input_key],
                    sql_cmd=sql_cmd,
                    error=truncated_error,
                )
                correct_error_prompt = populate_table_info_wrt_token_limit(
                    partial_error_prompt, self.llm, self.database, table_names_to_use
                )
                response = self.llm.generate_prompt(
                    [correct_error_prompt], callbacks=run_manager.get_child() if run_manager else None
                )
                sql_cmd = response.generations[0][0].text.strip()  # type: ignore

        if not verified:
            self.write_callback_message(
                f"Failed to verify SQL query after {self.max_retries} retries.", run_manager=run_manager, color="red"
            )
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {self.max_retries} tries.", sql_cmd
            )

        self.write_callback_message(sql_cmd, run_manager=run_manager, color="green")
        if get_config().enable_str_literal_correction:
            self.write_callback_message(
                "Attempting to correct string literals in SQL query.", run_manager=run_manager, color="blue"
            )
            sql_cmd = self.database.correct_string_literals(sql_cmd)
            self.write_callback_message(f"Result of correction: {sql_cmd}", run_manager=run_manager, color="green")

        sql_complexity = self.database.complexity(sql_cmd)

        return {self.output_key: strip_sql_comments(sql_cmd), self.output_complexity_key: str(sql_complexity)}

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        await self.write_callback_message_async(inputs[self.input_key], run_manager=run_manager)

        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")

        partial_gen_sql_prompt = self.prompt.partial(input=inputs[self.input_key])
        gen_sql_prompt = await run_in_threadpool(
            populate_table_info_wrt_token_limit, partial_gen_sql_prompt, self.llm, self.database, table_names_to_use
        )

        response = await self.llm.agenerate_prompt(
            [gen_sql_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        sql_cmd = response.generations[0][0].text.strip()  # type: ignore
        verified = False
        retries = 0

        while not verified:
            try:
                await self.write_callback_message_async(
                    f"Verifying SQL Query: {sql_cmd}", run_manager=run_manager, color="blue"
                )
                await run_in_threadpool(self.database.validate_query, sql_cmd)
                verified = True
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    break
                await self.write_callback_message_async(
                    f"Invalid SQL Query: {e}.", run_manager=run_manager, color="red"
                )
                def extract_error_message(error_str):
                    # Define the regular expression pattern to capture the inner error message
                    pattern = r'TDBException\(error_msg="(.+?)"\)'
                    
                    # Use the search method to find the pattern in the given error string
                    match = re.search(pattern, error_str)
                    
                    # If a match is found, extract the inner message. Otherwise, return the original string.
                    if match:
                        return match.group(1)
                    else:
                        return error_str
                extracted_error = extract_error_message(str(e))
                truncated_error = extracted_error[0:150] if len(extracted_error) > 150 else extracted_error
                partial_error_prompt = self.error_prompt.partial(
                    input=inputs[self.input_key],
                    sql_cmd=sql_cmd,
                    error=truncated_error,
                )
                correct_error_prompt = await run_in_threadpool(
                    populate_table_info_wrt_token_limit,
                    partial_error_prompt,
                    self.llm,
                    self.database,
                    table_names_to_use,
                )
                response = await self.llm.agenerate_prompt(
                    [correct_error_prompt], callbacks=run_manager.get_child() if run_manager else None
                )
                sql_cmd = response.generations[0][0].text.strip()  # type: ignore

        if not verified:
            await self.write_callback_message_async(
                f"Failed to verify SQL query after {self.max_retries} retries.", run_manager=run_manager, color="red"
            )
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {self.max_retries} tries.", sql_cmd
            )

        await self.write_callback_message_async(sql_cmd, run_manager=run_manager, color="green")

        if get_config().enable_str_literal_correction:
            await self.write_callback_message_async(
                "Attempting to correct string literals in SQL query.", run_manager=run_manager, color="blue"
            )
            sql_cmd = await run_in_threadpool(self.database.correct_string_literals, sql_cmd)
            await self.write_callback_message_async(
                f"Result of correction: {sql_cmd}", run_manager=run_manager, color="green"
            )

        sql_complexity = await run_in_threadpool(self.database.complexity, sql_cmd)

        return {self.output_key: strip_sql_comments(sql_cmd), self.output_complexity_key: str(sql_complexity)}

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chain"


class NLtoSQLChatChain(BaseNLtoSQLChain):
    llm: BaseChatModel
    prompt: BaseChatPromptTemplate = NL_TO_SQL_CHAT_PROMPT
    error_prompt: BasePromptTemplate = NL_TO_SQL_ERROR_PROMPT

    def get_sql_query(self, message: str) -> str:
        """
        Helps to parse out SQL query from the llm response message.
        """
        if "SQLQuery:" in message:
            return message.split("SQLQuery:")[1].strip()
        if "```sql" in message:
            return message.split("```sql")[1].strip()

        if ":\n" in message:
            return message.split(":\n")[1].strip()
        return strip_sql_comments(message)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        self.write_callback_message(inputs[self.input_key], run_manager=run_manager)

        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")

        partial_gen_sql_prompt = self.prompt.partial(input=inputs[self.input_key])
        gen_sql_prompt = populate_table_info_wrt_token_limit(
            partial_gen_sql_prompt, self.llm, self.database, table_names_to_use
        )

        messages = gen_sql_prompt.to_messages()
        response = self.llm.generate([messages], callbacks=run_manager.get_child() if run_manager else None)
        sql_cmd = self.get_sql_query(response.generations[0][0].text)
        verified = False
        retries = 0

        while not verified:
            try:
                messages.append(AIMessage(content=sql_cmd))
                self.write_callback_message(f"Verifying SQL Query: {sql_cmd}", run_manager=run_manager, color="blue")
                self.database.validate_query(sql_cmd)
                verified = True
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    break
                self.write_callback_message(f"Invalid SQL Query: {e}.", run_manager=run_manager, color="red")
                truncated_error = str(e)[0:150] if len(str(e)) > 150 else str(e)
                messages.append(HumanMessage(content=NL_TO_SQL_CHAT_ERROR_TEMPLATE.format(exception=truncated_error)))
                response = self.llm.generate([messages], callbacks=run_manager.get_child() if run_manager else None)
                sql_cmd = self.get_sql_query(response.generations[0][0].text)

        if not verified:
            self.write_callback_message(
                f"Failed to verify SQL query after {self.max_retries} retries.", run_manager=run_manager, color="red"
            )
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {self.max_retries} tries.", sql_cmd
            )

        self.write_callback_message(sql_cmd, run_manager=run_manager, color="green")

        if get_config().enable_str_literal_correction:
            self.write_callback_message(
                "Attempting to correct string literals in SQL query.", run_manager=run_manager, color="blue"
            )
            sql_cmd = self.database.correct_string_literals(sql_cmd)
            self.write_callback_message(f"Result of correction: {sql_cmd}", run_manager=run_manager, color="green")

        sql_complexity = self.database.complexity(sql_cmd)

        return {self.output_key: strip_sql_comments(sql_cmd), self.output_complexity_key: str(sql_complexity)}

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        await self.write_callback_message_async(inputs[self.input_key], run_manager=run_manager)

        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")

        partial_gen_sql_prompt = self.prompt.partial(input=inputs[self.input_key])
        gen_sql_prompt = await run_in_threadpool(
            populate_table_info_wrt_token_limit, partial_gen_sql_prompt, self.llm, self.database, table_names_to_use
        )

        messages = gen_sql_prompt.to_messages()
        response = await self.llm.agenerate([messages], callbacks=run_manager.get_child() if run_manager else None)
        sql_cmd = self.get_sql_query(response.generations[0][0].text)
        verified = False
        retries = 0

        while not verified:
            try:
                messages.append(AIMessage(content=sql_cmd))
                await self.write_callback_message_async(
                    f"Verifying SQL Query: {sql_cmd}", run_manager=run_manager, color="blue"
                )
                await run_in_threadpool(self.database.validate_query, sql_cmd)
                verified = True
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    break
                await self.write_callback_message_async(
                    f"Invalid SQL Query: {e}.", run_manager=run_manager, color="red"
                )
                truncated_error = str(e)[0:150] if len(str(e)) > 150 else str(e)
                messages.append(HumanMessage(content=NL_TO_SQL_CHAT_ERROR_TEMPLATE.format(exception=truncated_error)))
                response = await self.llm.agenerate(
                    [messages], callbacks=run_manager.get_child() if run_manager else None
                )
                sql_cmd = self.get_sql_query(response.generations[0][0].text)  # type: ignore

        if not verified:
            await self.write_callback_message_async(
                f"Failed to verify SQL query after {self.max_retries} retries.", run_manager=run_manager, color="red"
            )
            raise NLtoSQLException(
                f"Language model failed to generate a valid SQL query after {self.max_retries} tries.", sql_cmd
            )

        await self.write_callback_message_async(sql_cmd, run_manager=run_manager, color="green")

        if get_config().enable_str_literal_correction:
            await self.write_callback_message_async(
                "Attempting to correct string literals in SQL query.", run_manager=run_manager, color="blue"
            )
            sql_cmd = await run_in_threadpool(self.database.correct_string_literals, sql_cmd)
            await self.write_callback_message_async(
                f"Result of correction: {sql_cmd}", run_manager=run_manager, color="green"
            )

        sql_complexity = await run_in_threadpool(self.database.complexity, sql_cmd)

        return {self.output_key: strip_sql_comments(sql_cmd), self.output_complexity_key: str(sql_complexity)}

    @property
    def _chain_type(self) -> str:
        return "nl_to_sql_chat_chain"


def get_nl_to_sql_chain_by_llm(llm: BaseLanguageModel) -> type[BaseNLtoSQLChain]:
    """
    Gets the appropriate nt_to_sql chain class based upon the llm passed.
    """
    return NLtoSQLChatChain if isinstance(llm, BaseChatModel) else NLtoSQLChain  # type: ignore
