from typing import TypedDict

from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough

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


class SqlChainInputOutputDict(TypedDict):
    query: str
    inputs: SqlChainInputDict


class SqlChainOutputDict(TypedDict):
    query: str


class SqlChainOutputDictWithComplexity(TypedDict):
    query: str
    sql_complexity: int


# Chain to query with memory
async def get_table_info(sql_chain_inputs: SqlChainInputDict) -> str:
    heavydb = await get_db(sql_chain_inputs["session_id"])
    partial_gen_sql_prompt = get_value_from_runnable_binding(nl_to_sql_prompt_rbl).partial(input=sql_chain_inputs["question"])  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_sql_llm_rbl), heavydb, partial_gen_sql_prompt, sql_chain_inputs["tables"]  # type: ignore
    )
    return table_info


async def validate_query(input_output: SqlChainInputOutputDict) -> SqlChainInputOutputDict:
    """
    Validate the generated SQL query against HeavyDB.
    """
    query = input_output.get("query").strip()
    heavydb = await get_db(input_output["inputs"]["session_id"])
    await heavydb.avalidate_query(query)
    return input_output


async def do_string_literal_correction(input_output: SqlChainInputOutputDict) -> SqlChainInputOutputDict:
    """
    Do string literal correction on the generated SQL query.
    """
    if get_config().enable_str_literal_correction:
        input_output_copy = input_output.copy()
        query = input_output.get("query").strip()
        heavydb = await get_db(input_output["inputs"]["session_id"])
        corrected_query = await heavydb.acorrect_string_literals(query)
        input_output_copy["query"] = corrected_query
        return input_output_copy
    return input_output


async def calculate_sql_complexity(input_output: SqlChainInputOutputDict) -> SqlChainOutputDictWithComplexity:
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

# Chain which does the validation and string literal correction on the
# SQL query generated on the previous step (ie, by `query_runnable`)
chain = (
    RunnableParallel(inputs=input_runnable, query=query_runnable)
    | RunnableLambda(validate_query)  # type: ignore
    | RunnableLambda(do_string_literal_correction)  # type: ignore
    | RunnableLambda(lambda x: x["query"])  # type: ignore
).with_config(  # type: ignore
    config={"tags": ["nl_to_sql_chain_runnable"], "run_name": "nl_to_sql_chain_runnable"}  # type: ignore
)

chain_with_sql_complexity = (
    (RunnableParallel(inputs=input_runnable, query=chain) | RunnableLambda(calculate_sql_complexity))  # type: ignore
    .with_types(output_type=SqlChainOutputDictWithComplexity)  # type: ignore
    .with_config(  # type: ignore
        config={"tags": ["nl_to_sql_chain_with_sql_complexity_runnable"], "run_name": "nl_to_sql_chain_with_sql_complexity_runnable"}  # type: ignore
    )
)  # type: ignore
