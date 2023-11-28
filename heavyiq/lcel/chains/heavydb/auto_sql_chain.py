# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain.schema.runnable import Runnable, RunnablePassthrough

from heavyiq.lcel.types import AutoSQLChainInputType, AutoSQLChainOutputType

from .sql_chain import chain as sql_chain
from .table_chain import tables_list_chain

chain: Runnable = (RunnablePassthrough().assign(tables=tables_list_chain) | sql_chain).with_types(
    input_type=AutoSQLChainInputType, output_type=AutoSQLChainOutputType  # type: ignore
)
