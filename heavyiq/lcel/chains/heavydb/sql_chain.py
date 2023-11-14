from operator import itemgetter
from typing import Any, TypedDict

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_sql_prompt_runnable
from heavyiq.lcel.types.sql_type import (
    SqlChainInputType,
    SqlChainIntermediateDict,
    SqlChainIntermediateType,
    SqlChainOutputType,
)

# var endswith `rbl` means it's an runnable
nl_to_sql_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql", "llm_temperature": 0}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
)
nl_to_sql_prompt_rbl = (
    to_sql_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_sql_prompt_runnable
)
nl_to_sql_retry_prompt_rbl = (
    to_sql_prompt_runnable.with_config(configurable={"prompt": "custom_error"})
    if is_using_custom_trained_llm()
    else to_sql_prompt_runnable.with_config(configurable={"prompt": "openai_error"})
)


async def get_table_info(sql_chain_inputs: dict, on_retry: bool = False) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """
    prompt_rbl = nl_to_sql_retry_prompt_rbl if on_retry else nl_to_sql_prompt_rbl
    partial_inputs = {"input": sql_chain_inputs["question"]}
    if on_retry:
        partial_inputs.update({"sql_cmd": sql_chain_inputs["sql_cmd"], "error": sql_chain_inputs["error"]})  # type: ignore

    heavydb = await get_db(sql_chain_inputs["session_id"])
    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl).partial(**partial_inputs)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_sql_llm_rbl), heavydb, partial_gen_sql_prompt, sql_chain_inputs["tables"]  # type: ignore
    )
    return table_info


# runnable which stores the user input (ie. str or dict passed to runnable_chain.invoke method)
input_runnable = RunnablePassthrough().with_types(input_type=SqlChainInputType)  # type: ignore
# Supposed to return the SQL query predicted by the llm
query_runnable = (
    (
        input_runnable
        | RunnablePassthrough.assign(
            table_info=RunnableLambda(get_table_info), input=lambda x: x["question"]  # type: ignore
        )
        | nl_to_sql_prompt_rbl
        | nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>"])
        | StrOutputParser()
    )
    .with_config(config={"tags": ["nl_to_sql_predict_query_runnable"], "run_name": "Find Query"})  # type: ignore
    .with_types(input_type=SqlChainInputType)  # type: ignore
)


# SQL retry runnable

retry_input_runnable = RunnablePassthrough().with_types(input_type=SqlChainIntermediateType)  # type: ignore

retry_query_runnable = (
    retry_input_runnable
    | RunnablePassthrough.assign(
        table_info=RunnableLambda(get_table_info),  # type: ignore
    )
    | nl_to_sql_retry_prompt_rbl
    | nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>"])
    | StrOutputParser()
).with_config(
    config={"tags": ["nl_to_sql_retry_predict_query_runnable"], "run_name": "Query Retry"}  # type: ignore
)


async def sql_validator(input_output: dict) -> str | None:
    query = input_output["sql_cmd"].strip()
    session_id = input_output["session_id"]
    heavydb = await get_db(session_id=session_id)
    try:
        await heavydb.avalidate_query(query)
    except Exception as e:
        return str(e)

    return None


validation_step = (
    RunnablePassthrough()
    .assign(error=RunnableLambda(sql_validator))
    .with_config(config={"run_name": "SQL Query Validation"})
)


def revise_loop(input: SqlChainIntermediateDict) -> Runnable:
    revise_step = RunnablePassthrough().assign(sql_cmd=retry_query_runnable)

    else_step: Runnable[SqlChainIntermediateType, SqlChainIntermediateType] = RunnableBranch(
        (lambda x: x["error"] is None, RunnablePassthrough()),
        revise_step | validation_step,
    ).with_types(input_type=SqlChainIntermediateType)

    for _ in range(max(0, input["max_revisions"] - 1)):
        else_step = RunnableBranch(
            (lambda x: x["error"] is None, RunnablePassthrough()),
            revise_step | validation_step | else_step,
        )
    return else_step


revise_lambda = RunnableLambda(revise_loop).with_config(config={"run_name": "Revice Steps"})


async def do_string_literal_correction(inputs: dict) -> dict:
    """
    Do string literal correction on the generated SQL query.
    """
    sql_cmd = inputs["sql_cmd"].strip()
    if get_config().enable_str_literal_correction:
        heavydb = await get_db(inputs["session_id"])
        corrected_query = await heavydb.acorrect_string_literals(sql_cmd)
        sql_cmd = corrected_query

    return sql_cmd


async def calculate_sql_complexity(inputs: dict) -> int:
    """
    Calculate SQL complexity for the generated SQL query.
    """
    query = inputs["query"].strip()
    heavydb = await get_db(inputs["session_id"])
    sql_complexity = await heavydb.acomplexity(query)
    return sql_complexity


string_literal_correction_step = (
    RunnablePassthrough()
    .assign(query=RunnableLambda(do_string_literal_correction))
    .with_config(config={"run_name": "String Literal Correction"})
)

calculate_sql_complexity_step = (
    RunnablePassthrough()
    .assign(sql_complexity=RunnableLambda(calculate_sql_complexity))
    .with_config(config={"run_name": "Calculate SQL Complexity"})
)

final_step = RunnableLambda(lambda x: {"query": x["query"], "sql_complexity": x["sql_complexity"]})

chain: Runnable[Any, Any] = (
    (
        {
            "max_revisions": lambda x: 3,  # hardcoded retry count
            "session_id": itemgetter("session_id"),
            "question": itemgetter("question"),
            "tables": itemgetter("tables"),
        }
        | RunnablePassthrough().assign(sql_cmd=query_runnable)
        | validation_step
        | revise_lambda
        | string_literal_correction_step
        | calculate_sql_complexity_step
        | final_step
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoSQLChainRunnable"], "run_name": "NL to SQL Chain Runnable"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlChainOutputType)  # type: ignore
)
