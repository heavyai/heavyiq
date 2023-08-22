from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

from pydantic import Extra

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.schema import BasePromptTemplate, LLMResult
from langchain.prompts.prompt import PromptTemplate

from heavyiq.langchain.exceptions import GenerateTableMetadataException
from heavyiq.langchain.chains import BaseChain
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.utils import populate_table_info_wrt_token_limit

if TYPE_CHECKING:
    from heavydb._parsers import ColumnDetails

TABLE_METADATA_TEMPLATE = """
With respect to the SQL table schema and sample data provided,
please create a comprehensive response:
{table_info}

{format_instructions}
"""
TABLE_METADATA_PROMPT = PromptTemplate.from_template(TABLE_METADATA_TEMPLATE)


class GenerateTableMetadataChain(BaseChain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate = TABLE_METADATA_PROMPT
    database: HeavyDB
    llm: BaseLanguageModel | BaseChatModel
    input_key: str = "table_name"
    output_summary_key: str = "summary"  #: :meta private:
    output_columns_key: str = "columns"  #: :meta private:

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
        """Returns summary key. Column description key is not listed here because
        it is a dict of strings and chains expect all input and output keys to be strings.

        :meta private:
        """
        return [self.output_summary_key]

    def create_output_parser(self, columns: list[ColumnDetails]) -> StructuredOutputParser:
        response_schemas = [
            ResponseSchema(
                name="description",
                description="a brief yet informative summary of the table's purpose and primary and secondary functionality",
            )
        ]
        for column in columns:
            response_schemas.append(
                ResponseSchema(
                    name=column.name, description=f"a brief description of the {column.name} column's purpose"
                )
            )
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def clean_llm_response(self, response: LLMResult) -> str:
        # Removing backticks followed by the word "json"
        cleaned_text = re.sub(r"```json", "", response.generations[0][0].text)
        # Removing backticks alone
        cleaned_text = re.sub(r"```", "", cleaned_text)
        return cleaned_text

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        table_name = inputs[self.input_key]
        self.write_callback_message(f"Generating table metadata prompt for {table_name}")

        columns = self.database.get_table_columns(table_name)
        output_parser = self.create_output_parser(columns)
        gen_metadata_partial_prompt = self.prompt.partial(format_instructions=output_parser.get_format_instructions())
        gen_metadata_prompt = populate_table_info_wrt_token_limit(
            gen_metadata_partial_prompt, self.llm, self.database, [table_name]
        )

        self.write_callback_message("Calling LLM")
        response = self.llm.generate_prompt(
            [gen_metadata_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        try:
            self.write_callback_message("Parsing LLM response")
            parsed_response = output_parser.parse(self.clean_llm_response(response))
        except Exception as e:
            self.write_callback_message(f"Error parsing LLM response: {e}", color="red")
            raise GenerateTableMetadataException("Language model returned unparseable response. Please try again.")

        output = {self.output_summary_key: parsed_response["description"], self.output_columns_key: {}}
        for column in columns:
            output[self.output_columns_key][column.name] = parsed_response[column.name]

        return output

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        table_name = inputs[self.input_key]
        await self.write_callback_message_async(f"Generating table metadata prompt for {table_name}")

        columns = self.database.get_table_columns(table_name)
        output_parser = self.create_output_parser(columns)
        gen_metadata_partial_prompt = self.prompt.partial(format_instructions=output_parser.get_format_instructions())
        gen_metadata_prompt = populate_table_info_wrt_token_limit(
            gen_metadata_partial_prompt, self.llm, self.database, [table_name]
        )

        await self.write_callback_message_async("Calling LLM")
        response = self.llm.generate_prompt(
            [gen_metadata_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        try:
            await self.write_callback_message_async("Parsing LLM response")
            parsed_response = output_parser.parse(self.clean_llm_response(response))
        except Exception as e:
            await self.write_callback_message_async(f"Error parsing LLM response: {e}", color="red")
            raise GenerateTableMetadataException("Language model returned unparseable response. Please try again.")

        output = {self.output_summary_key: parsed_response["description"], self.output_columns_key: {}}
        for column in columns:
            output[self.output_columns_key][column.name] = parsed_response[column.name]

        return output

    @property
    def _chain_type(self) -> str:
        return "generate_table_metadata_chain"
