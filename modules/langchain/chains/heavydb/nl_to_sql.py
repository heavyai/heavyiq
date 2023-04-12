from __future__ import annotations
from typing import Any

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel
from pydantic import BaseModel, Extra, Field
from langchain.prompts.prompt import PromptTemplate

from modules.langchain.heavydb import HeavyDB

NL_TO_SQL_TEMPLATE = """Create a syntactically correct {dialect} query to answer the input question.
Only query relevant columns, avoiding SELECT * for any table.
Use only existing column names from the schema description, ensuring they are from the correct table.
Do not use STRING_AGG, GROUP_CONCAT functions.
Do not use STRING_AGG, GROUP_CONCAT functions.
Do not use STRING_AGG, GROUP_CONCAT functions.
Do not use STRING_AGG, GROUP_CONCAT functions.
Use GROUP BY if the question can be answered by aggregating over a column
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Explain your thinking step-by-step in a block comment before the query. Provide your response using the format below:
Question: "Question here"
SQLQuery: "/* step-by-step reasoning */ SQL Query to run"
Only use the tables listed below.
{table_info}
Question: {input}"""
NL_TO_SQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=NL_TO_SQL_TEMPLATE,
)

NL_TO_SQL_ERROR_TEMPLATE = """Correct the given {dialect} query:
If a function signature does not exist, do not use it. Reformulate the query to not use that function signature.
Do not use STRING_AGG, GROUP_CONCAT functions.
Do not use STRING_AGG, GROUP_CONCAT functions.
Do not use STRING_AGG, GROUP_CONCAT functions.
Use the following format:
Question: "Question here"
SQLQuery: "/* step-by-step thought process */ SQL Query to run"
Error: "Error message"
NewSQLQuery: "/* new step-by-step thought process */ Fixed SQL Query"
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
Errors: {error}
"""
NL_TO_SQL_ERROR_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "sql_cmd", "error"],
    template=NL_TO_SQL_ERROR_TEMPLATE,
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
    prompt: BasePromptTemplate = NL_TO_SQL_PROMPT
    """Prompt to use to translate natural language to SQL."""
    error_prompt: BasePromptTemplate = NL_TO_SQL_ERROR_PROMPT
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

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        error_recovery_chain = LLMChain(llm=self.llm, prompt=self.error_prompt)
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        self.callback_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables available to HeavyDB wrapper instance
        table_names_to_use = inputs.get("tables")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
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
                self.callback_manager.on_text(f"Verifying SQL Query: {sql_cmd}", color="blue", verbose=self.verbose)
                self.database.validate_query(sql_cmd)
                verified = True
            except Exception as e:
                self.callback_manager.on_text(f"Invalid SQL Query: {e}.", color="red", verbose=self.verbose)
                retry_llm_inputs = {
                    "input": input_text,
                    "sql_cmd": sql_cmd,
                    "error": f"{str(e)} \nNewSQLQuery:",
                    "dialect": self.database.dialect,
                    "table_info": table_info,
                    "stop": ["\nNewSQLQuery:"],
                }
                sql_cmd = error_recovery_chain.predict(**retry_llm_inputs)
                retries += 1
        if not verified:
            self.callback_manager.on_text(
                f"Failed to verify SQL query after {self.max_retries} retries.", color="red", verbose=self.verbose
            )
            raise Exception(f"Failed to verify SQL query after {self.max_retries} retries.")
        self.callback_manager.on_text(sql_cmd, color="green", verbose=self.verbose)

        chain_result: dict[str, Any] = {self.output_key: sql_cmd.strip()}
        return chain_result
