from typing import TypedDict

from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import HeavyDB
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_sql_prompt_runnable

# var endswith `rbl` means it's an runnable
nl_to_sql_llm_rbl = llm_runnable.with_config(configurable={"llm": "nl_to_sql", "llm_temperature": 0})
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


class SqlChainInputOutputDict(SqlChainInputDict):
    query: str


# Chain to query with memory
async def get_table_info(sql_chain_inputs: SqlChainInputDict) -> str:
    heavydb = await HeavyDB.from_session_async(sql_chain_inputs["session_id"])
    partial_gen_sql_prompt = get_value_from_runnable_binding(nl_to_sql_prompt_rbl).partial(input=sql_chain_inputs["question"])  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_sql_llm_rbl), heavydb, partial_gen_sql_prompt, sql_chain_inputs["tables"]  # type: ignore
    )
    return table_info


async def validate_query(input_output: SqlChainInputOutputDict) -> str:
    """
    Helps to validate the generated query against HeavyDB.
    """
    query = input_output.get("query").strip()
    heavydb = await HeavyDB.from_session_async(input_output["session_id"])
    await heavydb.avalidate_query(query)
    return query


sql_chain = (
    RunnablePassthrough.assign(
        table_info=RunnableLambda(get_table_info), input=lambda x: x["question"]  # type: ignore
    ).with_types(
        input_type=InputType  # type: ignore
    )
    | nl_to_sql_prompt_rbl
    | nl_to_sql_llm_rbl.bind(stop=["\nSQLResult:", "\n<|sql result|>"])
    | StrOutputParser()
)

chain = (RunnablePassthrough.assign(query=sql_chain).with_types(input_type=InputType) | validate_query).with_config(  # type: ignore
    config={"tags": ["nl_to_sql_chain_runnable"], "run_name": "nl_to_sql_chain_runnable"}  # type: ignore
)
