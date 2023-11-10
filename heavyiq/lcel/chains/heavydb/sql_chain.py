from typing import TypedDict

from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough

from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_sql_prompt_runnable
from heavyiq.lcel.types.sql_type import SqlChainInputType, SqlChainOutputType, SqlChainWithComplexityOutputType

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


class SqlQueryDict(TypedDict):
    query: str


class SqlChainInputWithQueryDict(TypedDict):
    query_dict: SqlQueryDict
    inputs: SqlChainInputDict


class ValidateQueryInputDict(TypedDict):
    query: str
    inputs: SqlChainInputDict


DetachedComplexityChainInputDict = ValidateQueryInputDict


class SqlChainOutputQueryDict(TypedDict):
    query: str


class SqlChainOutputQueryWithErrorDict(TypedDict):
    query: str
    error: str


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
    print(sql_chain_inputs)
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


async def validate_query(input_output: ValidateQueryInputDict) -> SqlChainOutputQueryDict:
    """
    Validate the generated SQL query against HeavyDB.
    """
    query = input_output["query"].strip()
    heavydb = await get_db(input_output["inputs"]["session_id"])
    try:
        await heavydb.avalidate_query(query)
    except Exception as e:
        raise NLtoSQLException(message=str(e), failed_sql=query)
    return {"query": query}


async def do_string_literal_correction(input_output: SqlChainInputWithQueryDict) -> SqlChainOutputQueryDict:
    """
    Do string literal correction on the generated SQL query.
    """
    query = input_output["query_dict"]["query"].strip()
    if get_config().enable_str_literal_correction:
        heavydb = await get_db(input_output["inputs"]["session_id"])
        corrected_query = await heavydb.acorrect_string_literals(query)
        query = corrected_query
    return {"query": query}


async def calculate_sql_complexity(input_output: SqlChainInputWithQueryDict) -> SqlChainOutputDictWithComplexity:
    """
    Calculate SQL complexity for the generated SQL query.
    """
    query = input_output["query_dict"]["query"].strip()
    heavydb = await get_db(input_output["inputs"]["session_id"])
    sql_complexity = await heavydb.acomplexity(query)
    return {"query": query, "sql_complexity": sql_complexity}


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
    .with_config(
        config={"tags": ["nl_to_sql_predict_query_runnable"], "run_name": "nl_to_sql_predict_query_runnable"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType)
)

# Chain which does the validation
chain_with_validation = RunnableParallel(inputs=input_runnable, query=query_runnable) | RunnableLambda(validate_query)  # type: ignore
chain = chain_with_validation.with_types(input_type=SqlChainInputType, output_type=SqlChainOutputType).with_config(  # type: ignore
    config={"tags": ["nl_to_sql_chain_runnable"], "run_name": "nl_to_sql_chain_runnable"}  # type: ignore
)

# Chain with string literal correction
chain_with_string_literal_correction = RunnablePassthrough.assign(
    inputs=input_runnable, query_dict=chain
) | RunnableLambda(
    do_string_literal_correction  # type: ignore
)

chain_with_sql_complexity = (
    (RunnableParallel(inputs=input_runnable, query_dict=chain_with_string_literal_correction) | RunnableLambda(calculate_sql_complexity))  # type: ignore
    .with_types(input_type=SqlChainInputType, output_type=SqlChainWithComplexityOutputType)  # type: ignore
    .with_config(  # type: ignore
        config={"tags": ["nl_to_sql_chain_with_sql_complexity_runnable"], "run_name": "nl_to_sql_chain_with_sql_complexity_runnable"}  # type: ignore
    )
)  # type: ignore

# chain which accepts query from user instead of predicting it using the above chain runnable
detached_complexity_chain = (
    (
        RunnablePassthrough.assign(
            query_dict=(
                RunnablePassthrough.assign(inputs=lambda x: x["inputs"], query_dict=lambda x: x["query"])  # type: ignore
                | RunnableLambda(do_string_literal_correction)  # type: ignore
            ),
            inputs=lambda x: x["inputs"],
        )
        | RunnableLambda(calculate_sql_complexity)  # type: ignore
    )
    .with_types(input_type=DetachedComplexityChainInputDict, output_type=SqlChainWithComplexityOutputType)  # type: ignore
    .with_config(  # type: ignore
        config={"tags": ["nl_to_sql_detached_complexity_chain"], "run_name": "nl_to_sql_detached_complexity_chain"}  # type: ignore
    )
)

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
retry_chain_with_validation = (
    RunnableParallel(inputs=retry_input_runnable, query=retry_query_runnable) | RunnableLambda(validate_query)  # type: ignore
).with_types(
    input_type=SqlRetryChainInputDict, output_type=SqlQueryDict  # type: ignore
)
retry_chain = retry_chain_with_validation
