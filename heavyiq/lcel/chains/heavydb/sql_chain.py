from typing import TypedDict

from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_sql_prompt_runnable

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


# # Supply the input types to the prompt
class InputType(BaseModel):
    question: str
    session_id: str
    tables: list[str]


# Supply the input types to
class SqlChainInputDict(TypedDict):
    question: str
    session_id: str
    tables: list[str]


class SqlChainInputWithQueryDict(TypedDict):
    query: str
    inputs: SqlChainInputDict


class SqlChainOutputDictWithComplexity(TypedDict):
    query: str
    sql_complexity: int


class SqlRetryChainInputDict(TypedDict):
    question: str
    session_id: str
    tables: list[str]
    sql_cmd: str
    error: str


async def get_table_info(sql_chain_inputs: SqlChainInputDict | SqlRetryChainInputDict, on_retry: bool = False) -> str:
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


async def validate_query(input_output: SqlChainInputWithQueryDict) -> str:
    """
    Validate the generated SQL query against HeavyDB.
    """
    # raise ValueError("Just a small exception.")
    query = input_output.get("query").strip()
    heavydb = await get_db(input_output["inputs"]["session_id"])
    await heavydb.avalidate_query(query)
    return query


async def do_string_literal_correction(input_output: SqlChainInputWithQueryDict) -> str:
    """
    Do string literal correction on the generated SQL query.
    """
    if get_config().enable_str_literal_correction:
        query = input_output.get("query").strip()
        heavydb = await get_db(input_output["inputs"]["session_id"])
        corrected_query = await heavydb.acorrect_string_literals(query)
        return corrected_query
    return input_output["query"]


async def calculate_sql_complexity(input_output: SqlChainInputWithQueryDict) -> SqlChainOutputDictWithComplexity:
    """
    Calculate SQL complexity for the generated SQL query.
    """
    query = input_output.get("query").strip()
    heavydb = await get_db(input_output["inputs"]["session_id"])
    sql_complexity = await heavydb.acomplexity(query)
    return {"query": query, "sql_complexity": sql_complexity}


# runnable which stores the user input (ie. str or dict passed to runnable_chain.invoke method)
input_runnable = RunnablePassthrough().with_types(input_type=SqlChainInputDict)  # type: ignore
# Supposed to return the SQL query predicted by the llm
query_runnable = (
    input_runnable
    | RunnablePassthrough.assign(
        table_info=RunnableLambda(get_table_info), input=lambda x: x["question"]  # type: ignore
    )
    | nl_to_sql_prompt_rbl
    | nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>"])
    | StrOutputParser()
).with_config(
    config={"tags": ["nl_to_sql_predict_query_runnable"], "run_name": "nl_to_sql_predict_query_runnable"}  # type: ignore
)

# Chain which does the validation
chain_with_validation = RunnableParallel(inputs=input_runnable, query=query_runnable) | RunnableLambda(validate_query)  # type: ignore

# Chain with string literal correction
# chain_with_string_literal_correction = RunnablePassthrough.assign(
#     inputs=input_runnable, query=chain_with_validation
# ) | RunnableLambda(
#     do_string_literal_correction  # type: ignore
# )

# chain = chain_with_string_literal_correction.with_config(  # type: ignore
#     config={"tags": ["nl_to_sql_chain_runnable"], "run_name": "nl_to_sql_chain_runnable"}  # type: ignore
# )
# chain_with_sql_complexity = (
#     (RunnableParallel(inputs=input_runnable, query=chain) | RunnableLambda(calculate_sql_complexity))  # type: ignore
#     .with_types(output_type=SqlChainOutputDictWithComplexity)  # type: ignore
#     .with_config(  # type: ignore
#         config={"tags": ["nl_to_sql_chain_with_sql_complexity_runnable"], "run_name": "nl_to_sql_chain_with_sql_complexity_runnable"}  # type: ignore
#     )
# )  # type: ignore


# SQL retry chains


retry_input_runnable = RunnablePassthrough().with_types(input_type=SqlRetryChainInputDict)  # type: ignore

retry_query_runnable = (
    retry_input_runnable
    | RunnablePassthrough.assign(
        table_info=RunnableLambda(get_table_info),  # type: ignore
    )
    | nl_to_sql_retry_prompt_rbl
    | nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>"])
    | StrOutputParser()
).with_config(
    config={"tags": ["nl_to_sql_retry_predict_query_runnable"], "run_name": "nl_to_sql_retry_predict_query_runnable"}  # type: ignore
)
retry_chain_with_validation = RunnableParallel(inputs=retry_input_runnable, query=retry_query_runnable) | RunnableLambda(validate_query)  # type: ignore


chain = RunnableBranch(
    (lambda x: "error" in x, retry_chain_with_validation), (lambda x: "error" not in x, chain_with_validation), lambda x: None  # type: ignore
).with_config(
    config={"tags": ["nl_to_sql_chain_runnable"], "run_name": "nl_to_sql_chain_runnable"}  # type: ignore
)


input_branch = RunnableBranch(
    (lambda x: "error" in x, retry_input_runnable), (lambda x: "error" not in x, input_runnable), lambda x: None  # type: ignore
)


chain_with_literal_correction = RunnableParallel(inputs=input_branch, query=chain) | RunnableLambda(do_string_literal_correction)  # type: ignore

chain_with_sql_complexity = (
    (RunnableParallel(inputs=input_branch, query=chain_with_literal_correction) | RunnableLambda(calculate_sql_complexity))  # type: ignore
    .with_types(output_type=SqlChainOutputDictWithComplexity)  # type: ignore
    .with_config(
        config={"tags": ["nl_to_sql_chain_with_sql_complexity_runnable"], "run_name": "nl_to_sql_chain_with_sql_complexity_runnable"}  # type: ignore
    )
)

complete_chain = chain_with_sql_complexity
