from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, Extra
from langchain.chat_models.base import BaseChatModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.prompt import PromptValue
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from heavyiq.langchain.heavydb import HeavyDB
from heavyiq.langchain.chains import BaseChain
from heavyiq.langchain.utils import populate_table_info_wrt_token_limit
from heavyiq.langchain.exceptions import OutputParserException


class CustomExpressionType(str, Enum):
    DIMENSION = "dimension"
    MEASURE = "measure"


class TableJoinInfo(BaseModel):
    join_type: str = Field(..., description="The type of join")
    left_table: str = Field(..., description="The left table")
    right_table: str = Field(..., description="The right table")
    left_join_key: str = Field(..., description="The left join key")
    right_join_key: str = Field(..., description="The right join key")
    left_database: str = Field(..., description="The left database")
    right_database: str = Field(..., description="The right database")


MEASURE_INSTRUCTIONS = """The statement this expression will be evaluated in will be grouped, so feel free to use aggregate functions.
If using count, feed free to use count(*) instead of a particular column.
These functions DO NOT EXIST: STRING_AGG, GROUP_CONCAT"""

DIMENSION_INSTRUCTIONS = (
    "The statement this expression will be evaluated in will not be grouped, so do not use aggregate functions."
)

JOIN_INSTRUCTIONS = """These tables are joined using a {join_type} join on the {left_table}.{left_join_key} and {right_table}.{right_join_key} columns.
Prepend the column name with the table name. For example, if you want to use the column "id" from the table "users", you would write "users.id"
"""

NOT_JOIN_INSTRUCTIONS = (
    "Do not prepend the column name with the table name, as the SQL statement may have been aliased."
)

CUSTOM_EXPRESSION_TEMPLATE = """
Create a syntactically correct SQL expression based on the provided question. As a reminder, a SQL expression is a combination of one or more values, operators and SQL functions that results in to a value and is NOT an entire SQL Statement.

{dim_or_measure_instructions}

Use only existing column names from the schema description, ensuring they are from the correct table.
DO NOT use double colon cast operators ("::"); instead use the CAST function if needed.
Explain your thinking step-by-step in a block comment before the query.
Only use the tables listed below.

{table_info}

{join_info}

Question: {query}

{format_instructions}
"""

response_schemas = [
    ResponseSchema(name="thought_process", description="Your thought process for the custom expression"),
    ResponseSchema(name="expression", description="The custom expression"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


class GenerateCustomExpressionChain(BaseChain):
    llm: BaseLanguageModel | BaseChatModel
    database: HeavyDB
    expression_type: CustomExpressionType
    table_name: str | None
    join_info: TableJoinInfo | None
    input_key: str = "query"  #: :meta private:
    output_key: str = "expression"  #: :meta private:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.table_name is None and self.join_info is None:
            raise ValueError("Must specify either table_name or join_info")

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
        return [self.output_key]

    def get_prompt(self, query: str) -> PromptValue:
        if self.expression_type == CustomExpressionType.MEASURE:
            dim_or_measure_instructions = MEASURE_INSTRUCTIONS
        else:
            dim_or_measure_instructions = DIMENSION_INSTRUCTIONS
        tables = []
        join_info = ""
        if self.table_name is not None:
            tables.append(self.table_name)
            join_info = NOT_JOIN_INSTRUCTIONS
        elif self.join_info is not None:
            tables.extend([self.join_info.left_table, self.join_info.right_table])
            join_info = JOIN_INSTRUCTIONS.format(
                join_type=self.join_info.join_type,
                left_table=self.join_info.left_table,
                right_table=self.join_info.right_table,
                left_join_key=self.join_info.left_join_key,
                right_join_key=self.join_info.right_join_key,
            )

        prompt_partial = PromptTemplate(
            template=CUSTOM_EXPRESSION_TEMPLATE,
            input_variables=["table_info"],
            partial_variables={
                "dim_or_measure_instructions": dim_or_measure_instructions,
                "query": query,
                "join_info": join_info,
                "format_instructions": output_parser.get_format_instructions(),
            },
        )

        return populate_table_info_wrt_token_limit(prompt_partial, self.llm, self.database, tables)

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        query = inputs[self.input_key]
        gen_custom_exp_prompt = self.get_prompt(query)
        self.write_callback_message(gen_custom_exp_prompt.to_string(), run_manager)
        response = self.llm.generate_prompt(
            [gen_custom_exp_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        try:
            parsed_response = output_parser.parse(response.generations[0][0].text)
            return parsed_response
        except Exception as e:
            self.write_callback_message(f"Error parsing LLM response: {e}", run_manager, color="red")
            raise OutputParserException("Language model returned unparseable response. Please try again.")

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        query = inputs[self.input_key]
        gen_custom_exp_prompt = self.get_prompt(query)
        await self.write_callback_message_async(gen_custom_exp_prompt.to_string(), run_manager, color="blue")
        response = await self.llm.agenerate_prompt(
            [gen_custom_exp_prompt], callbacks=run_manager.get_child() if run_manager else None
        )
        try:
            parsed_response = output_parser.parse(response.generations[0][0].text)
            return parsed_response
        except Exception as e:
            await self.write_callback_message_async(f"Error parsing LLM response: {e}", color="red")
            raise OutputParserException("Language model returned unparseable response. Please try again.")

    @property
    def _chain_type(self) -> str:
        return "custom_expression_chain"
