from __future__ import annotations

from typing import Any, Optional

from pydantic import Extra
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from heavyiq.langchain.chains import BaseChain
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.llms import is_using_custom_trained_llm
from .nl_to_sql import BaseNLtoSQLChain, get_nl_to_sql_chain_by_llm

ANSWER_TEMPLATE = """Given an input question, first create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
Use the following format:
Question: Question here
SQLQuery: /* step-by-step thought process */ SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
Question: {input}
SQLQuery: {sql_cmd}
SQLResult: {sql_result}"""
ANSWER_PROMPT = PromptTemplate.from_template(ANSWER_TEMPLATE)

CUSTOM_LLM_ANSWER_TEMPLATE = """<|english prompt|>
The user asked the following question:
{input}

To answer the question, the following SQL query was generated:
{sql_cmd}

The following results were returned:
{sql_result}

Now explain the results in English, referencing the question and the SQL query as needed.
<|english answer|>
"""
CUSTOM_LLM_ANSWER_PROMPT = PromptTemplate.from_template(CUSTOM_LLM_ANSWER_TEMPLATE)


class NLtoAnswerChain(BaseChain):
    """
    Chain for translating natural language to an answer.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    nl_sql_chain: BaseNLtoSQLChain
    database: HeavyDB
    input_key: str = "query"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    output_sql_key: str = "sql"  #: :meta private:
    output_sql_complexity_key: str = "sql_complexity"  #: :meta private:
    output_results_key: str = "results"  #: :meta private:
    output_fail_reason_key: str = "info"  #: :meta private:
    output_fail_reasons: dict[str, str] = {
        "max_size_reached": "Generated SQL query resultset exceeds the defined maximum result set size."
    }

    def __init__(self, *args, **kwargs):
        if is_using_custom_trained_llm():
            prompt = CUSTOM_LLM_ANSWER_PROMPT
        else:
            prompt = ANSWER_PROMPT
        super().__init__(*args, prompt=prompt, **kwargs)  # type: ignore

    @classmethod
    def from_same_llm(
        cls: type[NLtoAnswerChain], llm: BaseLanguageModel, database: HeavyDB, verbose: bool = False, **kwargs
    ) -> NLtoAnswerChain:
        """Create a new NLtoAnswerChain with the same LLM used for the NLtoSQLChain."""
        nl_sql_chain = get_nl_to_sql_chain_by_llm(llm)(llm=llm, database=database, verbose=verbose)
        return cls(llm=llm, nl_sql_chain=nl_sql_chain, database=database, verbose=verbose, **kwargs)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Will always return text key.

        :meta private:
        """
        return [
            self.output_key,
            self.output_results_key,
            self.output_sql_key,
            self.output_sql_complexity_key,
            self.output_fail_reason_key,
        ]

    def _should_forward_result_to_llm(self, sql_result: list) -> bool:
        """
        Based on the sql_result return a decision on whether to generate an answer using llm or not.
        """
        if not sql_result:
            # return True in case of empty result
            return True

        fields_count = len(sql_result) * len(sql_result[0])  # type: ignore
        if fields_count > get_config().max_fields_limit_sql_to_answer:
            return False

        return True

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        table_names_to_use = inputs.get("tables")
        nl_sql_inputs = {
            self.nl_sql_chain.input_key: inputs[self.input_key],
            "tables": table_names_to_use,
        }
        nl_sql_results = self.nl_sql_chain(nl_sql_inputs, callbacks=run_manager.get_child() if run_manager else None)
        sql_cmd = nl_sql_results[self.nl_sql_chain.output_key]
        self.write_callback_message(sql_cmd, run_manager=run_manager, color="green")

        sql_result = self.database.run(sql_cmd, to_str=False)
        pass_result_to_llm = self._should_forward_result_to_llm(sql_result)  # type: ignore
        sql_result = str(sql_result)

        self.write_callback_message("\nSQLResult: ", run_manager=run_manager, color="yellow")
        self.write_callback_message(sql_result, run_manager=run_manager, color="yellow")

        if pass_result_to_llm:
            gen_answer_prompt = self.prompt.format_prompt(
                input=inputs[self.input_key],
                sql_cmd=sql_cmd,
                sql_result=sql_result,
            )
            response = self.llm.generate_prompt(
                [gen_answer_prompt], callbacks=run_manager.get_child() if run_manager else None
            )
            answer, fail_reason = response.generations[0][0].text.strip(), ""  # type: ignore
            self.write_callback_message(f"\nAnswer:\n{answer}", run_manager=run_manager, color="green")
        else:
            answer, fail_reason = "", self.output_fail_reasons["max_size_reached"]
            self.write_callback_message(
                f"\nFailed to generate answer: {fail_reason}\nAnswer:\n{answer}",
                run_manager=run_manager,
                color="yellow",
            )

        return {
            self.output_key: answer,
            self.output_results_key: sql_result,
            self.output_sql_key: sql_cmd,
            self.output_sql_complexity_key: nl_sql_results[self.nl_sql_chain.output_complexity_key],
            self.output_fail_reason_key: fail_reason,
        }

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        table_names_to_use = inputs.get("tables")
        nl_sql_inputs = {
            self.nl_sql_chain.input_key: inputs[self.input_key],
            "tables": table_names_to_use,
        }
        nl_sql_results = await self.nl_sql_chain.acall(
            nl_sql_inputs, callbacks=run_manager.get_child() if run_manager else None
        )
        sql_cmd = nl_sql_results[self.nl_sql_chain.output_key]
        await self.write_callback_message_async(sql_cmd, run_manager=run_manager, color="green")

        sql_result = await self.database.arun(sql_cmd, to_str=False)
        pass_result_to_llm = self._should_forward_result_to_llm(sql_result)  # type: ignore
        sql_result = str(sql_result)

        await self.write_callback_message_async("\nSQLResult: ", run_manager=run_manager, color="yellow")
        await self.write_callback_message_async(sql_result, run_manager=run_manager, color="yellow")

        if pass_result_to_llm:
            gen_answer_prompt = self.prompt.format_prompt(
                input=inputs[self.input_key],
                sql_cmd=sql_cmd,
                sql_result=sql_result,
            )
            response = await self.llm.agenerate_prompt(
                [gen_answer_prompt], callbacks=run_manager.get_child() if run_manager else None
            )
            answer, fail_reason = response.generations[0][0].text.strip(), ""  # type: ignore
            await self.write_callback_message_async(f"\nAnswer:\n{answer}", run_manager=run_manager, color="green")
        else:
            answer, fail_reason = "", self.output_fail_reasons["max_size_reached"]
            await self.write_callback_message_async(
                f"\nFailed to generate answer: {fail_reason}\nAnswer:\n{answer}",
                run_manager=run_manager,
                color="yellow",
            )

        return {
            self.output_key: answer,
            self.output_results_key: sql_result,
            self.output_sql_key: sql_cmd,
            self.output_sql_complexity_key: nl_sql_results[self.nl_sql_chain.output_complexity_key],
            self.output_fail_reason_key: fail_reason,
        }

    @property
    def _chain_type(self) -> str:
        return "nl_to_answer_chain"
