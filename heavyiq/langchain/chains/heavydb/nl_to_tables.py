import re
from typing import Any, Dict, Optional
from pydantic import Extra
from langchain.schema import BasePromptTemplate
from heavyiq.langchain.heavydb import HeavyDB
from langchain.schema.language_model import BaseLanguageModel
from heavyiq.langchain.chains import BaseChain
from heavyiq.langchain.utils import apopulate_table_info_into_nl_to_tables_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun


NL_TO_TABLES_TEMPLATE = """
You are a experienced data analyst adept at analyzing user questions to determine the SQL tables required to generate an answer.
You have access to the following relational tables, with schemas below.

{table_info}

Emit each table name along with a 1 if the table is required to answer the following user question, or a 0 if the table is not required.

Question: {input}
Tables:"""


NL_TO_TABLES_PROMPT = PromptTemplate.from_template(NL_TO_TABLES_TEMPLATE)


class NLtoTablesChain(BaseChain):
    """
    Chain for analyzing natural language question to determine the SQL tables required to generate an answer.

    Note: You MUST restrict the tables that are used by either:
        1. Passing in a list of table names to use in the `tables` input key.
        2. Setting the `database` attribute to a HeavyDB object with the `include_tables` or `ignore_tables` attribute set.

        If you do not restrict the tables you risk exceeding the token limit of the LLM.
    """

    llm: BaseLanguageModel
    database: HeavyDB
    input_key: str = "question"  #: :meta private:
    output_key: str = "tables"  #: :meta private:
    prompt: BasePromptTemplate = NL_TO_TABLES_PROMPT

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        return "nl_to_tables_chain"

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

    def _call(self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun | None = None) -> Dict[str, Any]:
        raise NotImplementedError("Synchronous method not supported, use relevant async method.")

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, dict[str, int]]:
        question, table_names_to_use = inputs[self.input_key], inputs.get("tables")
        assert table_names_to_use, "Pass the list of table_names to consider."
        await self.write_callback_message_async(
            f"Finding relevant tables from {table_names_to_use} list for the nl question, {question}",
            run_manager=run_manager,
        )

        partial_gen_sql_prompt = self.prompt.partial(input=inputs[self.input_key])
        gen_table_prompt = await apopulate_table_info_into_nl_to_tables_prompt(
            partial_gen_sql_prompt, self.llm, self.database, table_names_to_use
        )

        response = await self.llm.agenerate_prompt(
            [gen_table_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        output = response.generations[0][0].text.strip()  # type: ignore
        required_tables = {x: int(y) for x, y in re.findall(r"(\w+):\s*([01])", output)}

        other_tables = {x: 0 for x in table_names_to_use if x not in required_tables.keys()}
        return {self.output_key: {**required_tables, **other_tables}}
