from typing import Any, TypedDict

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.lcel.chains.heavydb.sql_chain import SqlChainOutputDict
from heavyiq.lcel.chains.heavydb.sql_chain import chain as nl_to_sql_chain
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_answer_prompt_runnable

sql_to_answer_llm_rbl = llm_runnable.with_config(configurable={"llm": "sql_to_answer"})
sql_to_answer_prompt_rbl = (
    to_answer_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_answer_prompt_runnable
)


# Supply the input types to
class AnswerChainInputDict(TypedDict):
    question: str
    session_id: str
    tables: list[str]


class AnswerChainInputDictWithQuery(AnswerChainInputDict):
    # should be a verified/validated sql query. If you get the result from
    # sql_chain.chain runnable, then it should be a verified one.
    sql_cmd: str


async def get_sql_result(inputs: AnswerChainInputDictWithQuery) -> Any:
    """
    Gets the SQL query result by running the generated query against the HeavyDB.
    """
    sql_cmd = inputs["sql_cmd"]
    heavydb = await get_db(inputs["session_id"])
    sql_result = await heavydb.arun(sql_cmd, to_str=True)
    return sql_result


chain = (
    RunnablePassthrough.assign(sql_cmd=nl_to_sql_chain, input=lambda x: x["question"]).with_types(
        input_type=AnswerChainInputDict  # type: ignore
    )
    | RunnablePassthrough.assign(
        sql_result=RunnableLambda(get_sql_result),  # type: ignore
    )
    | sql_to_answer_prompt_rbl
    | sql_to_answer_llm_rbl
    | StrOutputParser()
).with_config(
    config={"tags": ["nl_to_answer_chain_runnable"], "run_name": "nl_to_answer_chain_runnable"}  # type: ignore
)
