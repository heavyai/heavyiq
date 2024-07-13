from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit, extract_error_message_from_exception
from heavyiq.lcel.chains.heavydb.relevant_info_chain import chain as relevant_info_chain
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_sql_prompt_runnable
from heavyiq.lcel.types.sql_type import (
    SqlChainInputType,
    SqlChainIntermediateDict,
    SqlChainIntermediateType,
    SqlChainOutputType,
)
from heavyiq.utils import strip_sql_comments

CONFIG = get_config()
# var endswith `rbl` means it's an runnable
nl_to_sql_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql", "llm_temperature": 0}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
)
nl_to_sql_error_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql_error", "llm_temperature": 0}, config={"tags": ["nl_to_sql_error_llm"]}  # type: ignore
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


async def get_table_info(sql_chain_inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """
    # if the inputs contain sql_cmd info then surely it's for retry prompt
    on_retry = True if sql_chain_inputs.get("sql_cmd") else False

    prompt_rbl = nl_to_sql_retry_prompt_rbl if on_retry else nl_to_sql_prompt_rbl
    partial_inputs = {"input": sql_chain_inputs["question"], "relevant_info": sql_chain_inputs.get("relevant_info", "")}
    if on_retry:
        partial_inputs.update({"sql_cmd": sql_chain_inputs["sql_cmd"], "relevant_info": sql_chain_inputs.get("relevant_info", ""), "error": sql_chain_inputs["error"]})  # type: ignore

    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl).partial(**partial_inputs)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_sql_llm_rbl), sql_chain_inputs["session_id"], partial_gen_sql_prompt, sql_chain_inputs["tables"]  # type: ignore
    )
    return table_info


# Step 1a
# calculating relevant info (RAG)
relevant_info_lambda: Runnable = RunnableBranch(
    (lambda x: not (CONFIG.custom_prompt_nl_to_sql_include_relevant_info), lambda x: ""),
    (lambda x: x.get("pre_calculated_relevant_info"), lambda x: x.get("pre_calculated_relevant_info")),
    (lambda x: CONFIG.custom_prompt_nl_to_sql_include_relevant_info, relevant_info_chain),
    lambda x: "",
).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Get Relevant Info for NLtoSQL",
        "metadata": {"step": "Getting relevant information based on the asked question."},
    }
)

# Step 1b
# Predicting Query
table_info_runnable_lambda: Runnable = RunnableLambda(get_table_info).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Get Table Info",
        "metadata": {"step": "Retrieving table information for prompt."},
    }
)

# Step 1
query_variables = (
    RunnablePassthrough.assign(relevant_info=relevant_info_lambda)
    | RunnablePassthrough.assign(table_info=table_info_runnable_lambda, input=lambda x: x["question"])
).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calculate Input Variables",
        "metadata": {"step": "Calculating input variables for Query prompt."},
    }
)

# Step 2
query_prompt: Runnable = nl_to_sql_prompt_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Preparing NL-SQL Prompt",
        "metadata": {"step": "Constructing prompt with input variables."},
    }
)
# Step 3
query_llm: Runnable = nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>", "\n<|sql answer|>"]).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calling LLM",
        "metadata": {"step": "Invoking the LLM for SQL query prediction."},
    }
)


# Supposed to return the SQL query predicted by the llm
query_runnable = (
    (query_variables | query_prompt | query_llm | StrOutputParser())
    .with_config(config={"tags": ["nl_to_sql_predict_query_runnable"], "run_name": "Find Query"})  # type: ignore
    .with_types(input_type=SqlChainInputType)  # type: ignore
)

# Query retry runnable which accepts error and sql_cmd from previous query prediction chain
# Retry Query Prediction
# Step 1
retry_query_variables = (RunnablePassthrough.assign(relevant_info=relevant_info_lambda) | RunnablePassthrough.assign(table_info=table_info_runnable_lambda, input=lambda x: x["question"])).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calculate Input Variables",
        "metadata": {"step": "Calculating input variables for Retry Query prompt."},
    }
)
# Step 2
retry_query_prompt: Runnable = nl_to_sql_retry_prompt_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Preparing NL-SQL Error Prompt",
        "metadata": {"step": "Constructing error prompt with input variables."},
    }
)

# Step 3
retry_query_llm: Runnable = nl_to_sql_error_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>", "\n<|sql error answer|>"]).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "ReCalling LLM",
        "metadata": {"step": "Invoking the LLM for SQL query retry prediction."},
    }
)

# SQL retry runnable
retry_query_runnable = (retry_query_variables | retry_query_prompt | retry_query_llm | StrOutputParser()).with_config(
    config={"tags": ["nl_to_sql_retry_predict_query_runnable"], "run_name": "Query Retry"}  # type: ignore
)


async def sql_validator(input_output: dict) -> str | None:
    """
    Validates SQL query against HeavyDB database.

    Args:
        input_output: inputs dict

    Returns:
        None for successfull validation or error string
    """
    query = input_output["sql_cmd"].strip()
    session_id = input_output["session_id"]
    heavydb = await get_db(session_id=session_id)
    try:
        await heavydb.avalidate_query(query)
    except Exception as e:
        return extract_error_message_from_exception(e)

    return None


validation_step = (
    RunnablePassthrough()
    .assign(error=RunnableLambda(sql_validator))
    .with_config(
        config={
            "run_name": "SQL Query Validation",
            "tags": ["intermediate-step"],
            "metadata": {"step": "Validating SQL query."},
        }
    )
)


async def revise_loop(input: SqlChainIntermediateDict) -> Runnable:
    """
    Iterates over bunch of steps to find a valid HeavyDB compatible SQL query.
    """
    revise_step = RunnablePassthrough().assign(sql_cmd=retry_query_runnable)

    else_step: Runnable[SqlChainIntermediateType, SqlChainIntermediateType] = RunnableBranch(
        (lambda x: x["error"] is None, RunnablePassthrough()),
        revise_step | validation_step,
    ).with_types(input_type=SqlChainIntermediateType)

    for _ in range(max(0, input["max_revisions"] - 1)):
        else_step = RunnableBranch(
            (lambda x: x["error"] is None, RunnablePassthrough()), (revise_step | validation_step | else_step)
        )
    return else_step


revise_lambda: Runnable = RunnableLambda(revise_loop)


async def do_string_literal_correction(inputs: dict) -> dict:
    """
    Do string literal correction on the generated SQL query.
    """
    sql_cmd = inputs["sql_cmd"].strip()
    if CONFIG.enable_str_literal_correction:
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
    .with_config(
        config={
            "run_name": "String Literal Correction",
            "tags": ["intermediate-step"],
            "metadata": {"step": "Correcting string literals on SQL Query."},
        }
    )
)

calculate_sql_complexity_step = (
    RunnablePassthrough()
    .assign(sql_complexity=RunnableLambda(calculate_sql_complexity))
    .with_config(
        config={
            "run_name": "Calculate SQL Complexity",
            "tags": ["intermediate-step"],
            "metadata": {"step": "Calculating SQL complexity."},
        }
    )
)

final_step = RunnableLambda(
    lambda x: {
        "query": strip_sql_comments(x.get("query", x.get("sql_cmd"))),
        "sql_complexity": x.get("sql_complexity", 0),
        "error": x["error"] or "",
    }
).with_config(
    config={
        "run_name": "Format Output",
        "tags": ["intermediate-step"],
        "metadata": {"step": "Formatting output."},
    }
)

# branch which passthrough the inputs upon error else do correct string literals and calculate complexity
do_string_correction_and_calculate_complexity_or_passthrough_branch: Runnable = RunnableBranch(
    (lambda x: x["error"] is None, string_literal_correction_step | calculate_sql_complexity_step),
    RunnablePassthrough(),
).with_types(input_type=SqlChainIntermediateType)

chain: Runnable[Any, Any] = (
    (
        RunnablePassthrough().assign(sql_cmd=query_runnable, max_revisions=lambda x: CONFIG.max_retries_nl_to_sql)
        | validation_step
        | revise_lambda
        | do_string_correction_and_calculate_complexity_or_passthrough_branch
        | final_step
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoSQLChainRunnable"], "run_name": "NL to SQL Chain Runnable"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlChainOutputType)  # type: ignore
)
