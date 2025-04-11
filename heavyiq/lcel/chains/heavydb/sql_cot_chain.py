from typing import Any



from langchain_core.runnables import Runnable

from langchain_core.runnables import RunnableBranch

from langchain_core.runnables import RunnableLambda

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from heavyiq.langchain.heavydb import get_config
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.output_parsers import COTJsonOutputParser
from heavyiq.lcel.prompts import to_sql_with_cot_prompt_runnable
from heavyiq.lcel.types.sql_type import SqlChainWithCOTOutputType
from heavyiq.utils import strip_sql_comments

from .sql_chain import (
    SqlChainInputType,
    do_string_correction_and_calculate_complexity_or_passthrough_branch,
    validation_step,
)

nl_to_sql_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql", "llm_temperature": 0, "llm_max_tokens": 1024}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
)
nl_to_sql_error_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql_error", "llm_temperature": 0}, config={"tags": ["nl_to_sql_error_llm"]}  # type: ignore
)
nl_to_sql_cot_prompt_rbl = (
    to_sql_with_cot_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_sql_with_cot_prompt_runnable
)
nl_to_sql_cot_retry_prompt_rbl = (
    to_sql_with_cot_prompt_runnable.with_config(configurable={"prompt": "custom_error"})
    if is_using_custom_trained_llm()
    else to_sql_with_cot_prompt_runnable.with_config(configurable={"prompt": "openai_error"})
)


async def get_table_info(sql_chain_inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """
    # if the inputs contain sql_cmd info then surely it's for retry prompt
    on_retry = True if sql_chain_inputs.get("sql_cmd") else False

    prompt_rbl = nl_to_sql_cot_retry_prompt_rbl if on_retry else nl_to_sql_cot_prompt_rbl
    partial_inputs = {"input": sql_chain_inputs["question"]}
    if on_retry:
        partial_inputs.update({"sql_cmd": sql_chain_inputs["sql_cmd"], "error": sql_chain_inputs["error"]})  # type: ignore

    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl).partial(**partial_inputs)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_sql_llm_rbl), sql_chain_inputs["session_id"], partial_gen_sql_prompt, sql_chain_inputs["tables"]  # type: ignore
    )
    return table_info


table_info_runnable_lambda: Runnable = RunnableLambda(get_table_info).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Get Table Info",
        "metadata": {"step": "Retrieving table information for prompt."},
    }
)
query_variables = RunnablePassthrough.assign(table_info=table_info_runnable_lambda, input=lambda x: x["question"]).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calculate Input Variables",
        "metadata": {"step": "Calculating input variables for Query prompt."},
    }
)

# Step 2
query_prompt: Runnable = nl_to_sql_cot_prompt_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Preparing NL-SQL-COT Prompt",
        "metadata": {"step": "Constructing prompt with input variables."},
    }
)
# Step 3
query_llm: Runnable = nl_to_sql_llm_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "call_llm",
        "metadata": {"step": "Invoking the LLM for SQL query prediction."},
    }
)

query_cot_runnable = (
    (query_variables | query_prompt | query_llm.with_config({"run_name": "stream_llm"}) | COTJsonOutputParser())
    .with_config(config={"tags": ["nl_to_sql_cot_predict_query_runnable"], "run_name": "find_query"})  # type: ignore
    .with_types(input_type=SqlChainInputType)  # type: ignore
)

final_step = RunnableLambda(
    lambda x: {
        "query": strip_sql_comments(x.get("query", x.get("sql_cmd"))),
        "sql_complexity": x.get("sql_complexity", 0),
        "error": x["error"] or "",
        "cot": x["cot"],
    }
).with_config(
    config={
        "run_name": "Format Output",
        "tags": ["intermediate-step"],
        "metadata": {"step": "Formatting output."},
    }
)

# retry
# Step 1
retry_query_variables = RunnablePassthrough.assign(table_info=table_info_runnable_lambda, input=lambda x: x["question"]).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Retry: Calculate Input Variables",
        "metadata": {"step": "Calculating input variables for Retry Query prompt."},
    }
)
# Step 2
retry_query_prompt: Runnable = nl_to_sql_cot_retry_prompt_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Retry: Preparing NL-SQL Error Prompt",
        "metadata": {"step": "Constructing error prompt with input variables."},
    }
)
# Step 3
retry_query_llm = query_llm

# SQL with COT retry runnable
retry_cot_query_runnable = (
    retry_query_variables
    | retry_query_prompt
    | retry_query_llm.with_config({"run_name": "stream_retry_llm"})
    | StrOutputParser()
).with_config(
    config={"tags": ["nl_to_sql_cot_retry_query_runnable"], "run_name": "Query with COT Retry"}  # type: ignore
)


async def unpack_cot(inputs: dict) -> dict:
    if not inputs:
        return inputs
    sql_with_cot = inputs.pop("sql_with_cot", None)
    if sql_with_cot and isinstance(sql_with_cot, dict):
        return {**inputs, "sql_cmd": sql_with_cot["sql"], "cot": sql_with_cot["cot"]}
    return inputs


async def revise_loop(input: Any) -> Runnable:
    """
    Iterates over bunch of steps to find a valid HeavyDB compatible SQL query.
    """
    revise_step = RunnablePassthrough().assign(sql_cmd=retry_cot_query_runnable)

    else_step: Runnable[Any] = RunnableBranch(
        (lambda x: x["error"] is None, RunnablePassthrough()),
        revise_step | validation_step,
    )

    for _ in range(max(0, input["max_revisions"] - 1)):
        else_step = RunnableBranch(
            (lambda x: x["error"] is None, RunnablePassthrough()), (revise_step | validation_step | else_step)
        )
    return else_step


revise_lambda: Runnable = RunnableLambda(revise_loop)


chain: Runnable[Any, Any] = (
    (
        RunnablePassthrough().assign(sql_with_cot=query_cot_runnable)
        | RunnableLambda(unpack_cot)
        | RunnablePassthrough().assign(
            max_revisions=lambda x: get_config().max_retries_nl_to_sql,
        )
        | validation_step
        | revise_lambda
        | do_string_correction_and_calculate_complexity_or_passthrough_branch
        | final_step
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoSQLChainRunnable"], "run_name": "NL to SQL Chain Runnable"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlChainWithCOTOutputType)  # type: ignore
)
