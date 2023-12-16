# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.lcel.types import AutoAnswerChainInputType, AutoAnswerChainOutputType

from .answer_chain import nl_to_answer_chain as answer_chain
from .table_chain import tables_list_chain


def format_output(inputs: dict) -> dict:
    return {**inputs["output"], "tables": inputs["tables"]}


chain: Runnable = (
    RunnablePassthrough().assign(tables=tables_list_chain)
    | RunnablePassthrough().assign(output=answer_chain)
    | RunnableLambda(format_output)
).with_types(
    input_type=AutoAnswerChainInputType, output_type=AutoAnswerChainOutputType  # type: ignore
)
