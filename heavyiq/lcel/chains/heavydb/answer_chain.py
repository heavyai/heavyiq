from typing import Any, TypedDict

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.config import get_config
from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.lcel.chains.heavydb.sql_chain import calculate_sql_complexity
from heavyiq.lcel.chains.heavydb.sql_chain import chain as nl_to_sql_chain_with_sql_complexity
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
    ).dict()


def change_format_for_error(inputs: dict) -> Any:
    """
    Return this dict response for invalid query generation.
    """
    return AnswerChainOutputType(
        sql=inputs["query"],
        sql_complexity=inputs["sql_complexity"],
        results="",
        answer="",
        fail_reason=inputs["sql_chain_output"]["error"],
    ).dict()


generate_sql_resultset_step = RunnablePassthrough.assign(
    sql_result=RunnableLambda(get_sql_result),  # type: ignore
).with_config(config={"run_name": "Generate SQL Resultset"})
should_forward_resultset_to_llm_step = RunnablePassthrough.assign(
    do_forward=RunnableLambda(should_forward_sql_result_to_llm)
).with_config(config={"run_name": "Should Forward Resultset to LLM?"})

sql_to_answer_step = (sql_to_answer_prompt_rbl | sql_to_answer_llm_rbl | StrOutputParser()).with_config(
    config={"run_name": "Predict Answer"}
)

# branch which either generates the answer or returns None
answer_decider_branch = (
    RunnablePassthrough()
    .assign(
        answer=RunnableBranch(
            (lambda x: x["do_forward"], sql_to_answer_step),
            (lambda x: not x["do_forward"], RunnableLambda(lambda x: "")),
            lambda x: "",
        ),
        inputs=lambda x: x,
    )
    .with_config(config={"run_name": "Generate Answer or Return Empty Branch"})
)
# branch which checks for the existence of `query` and `sql_complexity` keys in input.
# If yes, then it forms the input for the next runnable step by assigning the values to `sql_chain_output`
# else it asks for nl_to_sql_chain_with_sql_complexity runnable to generate query and sql_complexity
to_sql_chain_or_not_branch = RunnableBranch(
    (
        lambda x: "query" in x and "sql_complexity" in x,
        RunnableLambda(lambda x: {"query": x["query"], "sql_complexity": x["sql_complexity"]}),  # type: ignore
    ),
    (
        lambda x: "query" in x and "sql_complexity" not in x,
        RunnableLambda(lambda x: {"query": x["query"], "sql_complexity": RunnableLambda(calculate_sql_complexity)}),  # type: ignore
    ),
    nl_to_sql_chain_with_sql_complexity,
).with_config(config={"run_name": "Find Query or Passthrough Branch"})

final_step = RunnableLambda(change_format)
final_error_step = RunnableLambda(change_format_for_error)
# Initial branch which will do the answer geenration only when the query
# gets validated successfully.
handle_query_error_branch: Runnable = RunnableBranch(
    (lambda x: bool(x["sql_chain_output"].get("error")), RunnablePassthrough() | final_error_step),
    RunnablePassthrough.assign(
        sql_cmd=lambda x: x["sql_chain_output"]["query"],
        sql_complexity=lambda x: x["sql_chain_output"]["sql_complexity"],
    )
    | generate_sql_resultset_step
    | should_forward_resultset_to_llm_step
    | answer_decider_branch
    | final_step,
)

# Chain of RunnablePassthrough merges all the intermediate results with the original input. For ex,
# RunnablePassthrough() | RunnablePassthrough.assign(foo=RunnableLambda(fun_a)) | RunnablePassthrough.assign(bar=RunnableLambda(func_b))
# here func_b function receives the original input as well as the intermediate results, ie. foo
chain = (
    (
        RunnablePassthrough.assign(sql_chain_output=to_sql_chain_or_not_branch, input=lambda x: x["question"])
        | handle_query_error_branch
    )
    .with_types(input_type=SqlChainInputType, output_type=AnswerChainOutputType)  # type: ignore
    .with_config(
        config={"tags": ["NLtoAnswerChainRunnable"], "run_name": "NL to Answer Chain Runnable"}  # type: ignore
    )
)
