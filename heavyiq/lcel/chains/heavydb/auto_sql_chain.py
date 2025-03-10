# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.


from langchain_core.runnables import Runnable

from langchain_core.runnables import RunnableBranch

from langchain_core.runnables import RunnableLambda

from langchain_core.runnables import RunnablePassthrough

from heavyiq.langchain.heavydb import get_config
from heavyiq.lcel.types import AutoSQLChainInputType, AutoSQLChainOutputType

from .relevant_info_chain import chain as relevant_info_chain
from .sql_chain import chain as sql_chain
from .table_chain import tables_list_chain


def format_output(inputs: dict) -> dict:
    return {**inputs["output"], "tables": inputs["tables"], "snippet_ids": inputs["snippet_ids"]}


CONFIG = get_config()
relevant_info_default_values = {"snippet_ids": [], "relevant_info": ""}

chain: Runnable = (
    RunnablePassthrough.assign(
        pre_calculated_relevant_info_dict=RunnableBranch(
            (lambda x: not (CONFIG.enable_rag), lambda x: relevant_info_default_values),
            (
                lambda x: CONFIG.custom_prompt_nl_to_tables_include_relevant_info
                or CONFIG.custom_prompt_nl_to_sql_include_relevant_info,
                relevant_info_chain,
            ),
            lambda x: relevant_info_default_values,
        )
    )
    | RunnablePassthrough.assign(
        snippet_ids=lambda x: x["pre_calculated_relevant_info_dict"]["snippet_ids"],
        relevant_info=lambda x: x["pre_calculated_relevant_info_dict"]["relevant_info"],
    )
    | RunnablePassthrough().assign(tables=tables_list_chain)
    | RunnablePassthrough().assign(output=(RunnableBranch((lambda x: x["tables"], sql_chain), lambda x: {})))
    | RunnableLambda(format_output).with_types(
        input_type=AutoSQLChainInputType, output_type=AutoSQLChainOutputType  # type: ignore
    )
)
