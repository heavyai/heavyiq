from __future__ import annotations
from typing import Any

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel
from pydantic import BaseModel, Extra, Field

from modules.langchain.heavydb import HeavyDB

from .nl_to_sql import NLtoSQLChain

ANSWER_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:
Question: "Question here"
SQLQuery: "/* step-by-step thought process */ SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
SQLResult: {sql_result}"""
ANSWER_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "sql_cmd", "sql_result"],
    template=ANSWER_TEMPLATE,
)


class NLtoAnswerChain(Chain, BaseModel):
    """
    Chain for translating natural language to an answer.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseLanguageModel
    """LLM wrapper to use."""
    database: HeavyDB = Field(exclude=True)
    """HeavyDB Database to connect to."""
    prompt: BasePromptTemplate = ANSWER_PROMPT
    """Prompt to use to translate natural language to SQL."""
    input_key: str = "query"  #: :meta private:
    output_answer_key: str = "answer"  #: :meta private:
    output_sql_key: str = "sql"  #: :meta private:
    output_results_key: str = "results"  #: :meta private:
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> list[str]:
        """Return the singular input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys.
        :meta private:
        """
        output_keys = [self.output_sql_key, self.output_results_key]
        if not self.return_direct:
            output_keys.append(self.output_answer_key)
        return output_keys

    @property
    def _chain_type(self) -> str:
        return "nl_to_answer_chain"

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        table_names_to_use = inputs.get("tables")
        nl_sql_chain = NLtoSQLChain(llm=self.llm, database=self.database, verbose=self.verbose)
        nl_sql_inputs = {
            nl_sql_chain.input_key: inputs[self.input_key],
            "tables": table_names_to_use,
        }
        nl_sql_results = nl_sql_chain(nl_sql_inputs)
        sql_cmd = nl_sql_results[nl_sql_chain.output_key]
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        self.callback_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
        result = self.database.run(sql_cmd)
        self.callback_manager.on_text("\nSQLResult: ", verbose=self.verbose)
        self.callback_manager.on_text(result, color="yellow", verbose=self.verbose)

        if self.return_direct:
            return {
                self.output_sql_key: sql_cmd,
                self.output_results_key: result,
            }
        else:
            self.callback_manager.on_text("\nAnswer:", verbose=self.verbose)
            llm_chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=self.verbose, output_key="answer")
            llm_inputs = {
                "input": inputs[self.input_key],
                "dialect": self.database.dialect,
                "sql_cmd": sql_cmd,
                "table_info": table_info,
                "sql_result": f"{result} \nAnswer:",
                "stop": ["\nAnswer:"],
            }
            final_result = llm_chain.predict(**llm_inputs)
            self.callback_manager.on_text(final_result, color="green", verbose=self.verbose)
            return {
                self.output_sql_key: sql_cmd,
                self.output_results_key: result,
                self.output_answer_key: final_result.strip(),
            }
