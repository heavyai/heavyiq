from typing import Any, TypedDict

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.config import get_config
from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.lcel.chains.heavydb.sql_chain import chain_with_sql_complexity as nl_to_sql_chain_with_sql_complexity
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_answer_prompt_runnable
from heavyiq.lcel.types.answer_type import AnswerChainOutputType
from heavyiq.lcel.types.sql_type import SqlChainInputType

sql_to_answer_llm_rbl = llm_runnable.with_config(configurable={"llm": "sql_to_answer"})
sql_to_answer_prompt_rbl = (
    to_answer_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_answer_prompt_runnable
)
# reasons about the fail
output_fail_reasons: dict[str, str] = {
    "max_size_reached": "Generated SQL query resultset exceeds the defined maximum result set size."
}


# Supply the input types to
class AnswerChainInputDict(TypedDict):
    question: str
    session_id: str
    tables: list[str]


class AnswerChainInputDictWithQuery(AnswerChainInputDict):
    # should be a verified/validated sql query. If you get the result from
    # sql_chain.chain runnable, then it should be a verified one.
    sql_cmd: str


async def get_sql_result(inputs: AnswerChainInputDictWithQuery) -> list | str | tuple:
    """
    Gets the SQL query result by running the generated query against the HeavyDB.
    """
    sql_cmd = inputs["sql_cmd"]
    heavydb = await get_db(inputs["session_id"])
    sql_result = await heavydb.arun(sql_cmd, to_str=False)
    return sql_result


def should_forward_sql_result_to_llm(inputs: dict) -> bool:
    """
    Based on the sql_result, return a decision on whether to generate an answer using llm or not.
    """
    sql_result = inputs["sql_result"]
    if not sql_result:
        # return True in case of empty result
        return True

    fields_count = len(sql_result) * len(sql_result[0])  # type: ignore
    if fields_count > get_config().max_fields_limit_sql_to_answer:
        return False

    return True


def change_format(inputs: dict) -> Any:
    """
    Helps to change answer output to a desired format.
    """
    return AnswerChainOutputType(
        sql=inputs["sql_cmd"],
        sql_complexity=inputs["sql_complexity"],
        results=str(inputs["sql_result"]),
        answer=inputs["answer"],
        fail_reason=""
        if inputs["do_forward"]
        else output_fail_reasons["max_size_reached"],  # Fix this for more than 1 fail reasons
    )


# branch which either generates the answer or returns None
branch = RunnablePassthrough().assign(
    answer=RunnableBranch(
        (lambda x: x["do_forward"], sql_to_answer_prompt_rbl | sql_to_answer_llm_rbl | StrOutputParser()),
        (lambda x: not x["do_forward"], RunnableLambda(lambda x: "")),
        lambda x: "",
    ),
    inputs=lambda x: x,
)
# Chain of RunnablePassthrough merges all the intermediate results with the original input. For ex,
# RunnablePassthrough() | RunnablePassthrough.assign(foo=RunnableLambda(fun_a)) | RunnablePassthrough.assign(bar=RunnableLambda(func_b))
# here func_b function receives the original input as well as the intermediate results, ie. foo
chain = (
    (
        RunnablePassthrough.assign(sql_chain_output=nl_to_sql_chain_with_sql_complexity, input=lambda x: x["question"])
        | RunnablePassthrough.assign(
            sql_cmd=lambda x: x["sql_chain_output"]["query"],
            sql_complexity=lambda x: x["sql_chain_output"]["sql_complexity"],
        )
        | RunnablePassthrough.assign()
        | RunnablePassthrough.assign(
            sql_result=RunnableLambda(get_sql_result),  # type: ignore
        )
        | RunnablePassthrough.assign(do_forward=RunnableLambda(should_forward_sql_result_to_llm))
        | branch
        | RunnableLambda(change_format)
    )
    .with_types(input_type=SqlChainInputType, output_type=AnswerChainOutputType)  # type: ignore
    .with_config(
        config={"tags": ["nl_to_answer_chain_runnable"], "run_name": "nl_to_answer_chain_runnable"}  # type: ignore
    )
)
