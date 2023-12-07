# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain.schema.runnable import Runnable, RunnablePassthrough

from heavyiq.lcel.types import AutoAnswerChainInputType, AutoAnswerChainOutputType

from .answer_chain import chain as answer_chain
from .table_chain import tables_list_chain

chain: Runnable = (RunnablePassthrough().assign(tables=tables_list_chain) | answer_chain).with_types(
    input_type=AutoAnswerChainInputType, output_type=AutoAnswerChainOutputType  # type: ignore
)
