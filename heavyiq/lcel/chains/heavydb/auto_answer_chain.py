# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.lcel.types import AutoAnswerChainInputType, AutoAnswerChainOutputType

from .answer_chain import chain as answer_chain
from .table_chain import tables_list_chain


def format_output(inputs: dict) -> dict:
    return {**inputs["output"], "tables": inputs["tables"]}


chain: Runnable = (
    RunnablePassthrough().assign(tables=tables_list_chain)
    | RunnablePassthrough().assign(output=RunnableBranch((lambda x: x["tables"], answer_chain), lambda x: {}))
    | RunnableLambda(format_output)
).with_types(
    input_type=AutoAnswerChainInputType, output_type=AutoAnswerChainOutputType  # type: ignore
)
