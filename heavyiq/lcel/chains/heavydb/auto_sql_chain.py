# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.lcel.types import AutoSQLChainInputType, AutoSQLChainOutputType
from heavyrag import IndexNotFound

from .sql_chain import chain as sql_chain
from .table_chain import tables_list_chain


def format_output(inputs: dict) -> dict:
    return {**inputs["output"], "tables": inputs["tables"]}


CONFIG = get_config()


async def get_relevant_info_using_rag(inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """
    from heavyrag.main import get_relevant_facts_info

    if not (
        CONFIG.custom_prompt_nl_to_tables_include_relevant_info or CONFIG.custom_prompt_nl_to_sql_include_relevant_info
    ):
        return ""

    heavydb = await get_db(inputs["session_id"])

    try:
        relevant_facts = await get_relevant_facts_info(
            question=inputs["question"],
            heavydb_name=heavydb._dbname,
            similarity_cutoff=CONFIG.rag_facts_similarity_cutoff_score,
            similarity_top_k=CONFIG.rag_facts_similarity_top_k,
            with_reranker=True,
            reranker_top_k=CONFIG.rag_facts_reranker_top_k,
            reranker_cutoff=CONFIG.rag_facts_reranker_cutoff_score,
        )
    except IndexNotFound:
        return ""

    if not relevant_facts:
        return ""
    # has relevant facts
    return f"Relevant info:\n\n{relevant_facts}\n"


relevant_info_lambda: Runnable = RunnableLambda(get_relevant_info_using_rag).with_config(
    config={
        "tags": ["intermediate-step"],
        "run_name": "Get Relevant Info for AutoSQL Chain",
        "metadata": {"step": "Getting relevant information based on the asked question."},
    }
)


chain: Runnable = (
    RunnablePassthrough.assign(pre_calculated_relevant_info=relevant_info_lambda)
    | RunnablePassthrough().assign(tables=tables_list_chain)
    | RunnablePassthrough().assign(output=RunnableBranch((lambda x: x["tables"], sql_chain), lambda x: {}))
    | RunnableLambda(format_output).with_types(
        input_type=AutoSQLChainInputType, output_type=AutoSQLChainOutputType  # type: ignore
    )
)
