# SQL Generations chain
# chain used to generate n SQL queries by turning off the beam search
import asyncio
from collections.abc import Sequence

from langchain.schema.runnable import RunnableLambda

from heavyiq.langchain.heavydb import heavydb_var
from heavyiq.langchain.utils import extract_error_message_from_exception
from heavyiq.lcel.chains.heavydb.sql_chain import query_prompt, query_variables
from heavyiq.lcel.chains.utils import configure_step
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.runnables.base import LLMGenerationRunnable, SessionRunnable

llm = llm_runnable.with_config(
    configurable={"llm": "nl_to_sql", "llm_temperature": 0.7, "llm_n": 5}, config={"tags": ["nl_to_sql_gen_llm"]}  # type: ignore
)
model = configure_step(LLMGenerationRunnable(llm), run_name="Call LLM", step="Calling LLM.")


async def validate(query: str) -> tuple[bool, str]:
    """
    Helps to validate a single sql query.
    """
    heavydb = heavydb_var.get()
    try:
        await heavydb.avalidate_query(query)
    except Exception as e:
        return False, extract_error_message_from_exception(e)
    else:
        return True, ""


async def sql_list_validator(sql_queries: Sequence[str]) -> tuple[bool, str]:
    """
    Validates list of SQL queries and picks the right one.
    """

    error = ""
    for query in sql_queries:
        success, error = await validate(query)
        if success:
            return True, query

    return False, error


async def valid_query_filter(sql_queries: Sequence[str]) -> Sequence[str]:
    """
    Helps to filter out valid queries from the list of queries.
    """
    tasks = []
    for query in sql_queries:
        tasks.append(validate(query))

    out = await asyncio.gather(*tasks)
    return [q for q, (success, _) in zip(sql_queries, out) if success]


validate_queries = RunnableLambda(sql_list_validator).with_config(
    config={
        "run_name": "SQL Query Validation",
        "tags": ["intermediate-step"],
        "metadata": {"step": "Validating SQL query."},
    }
)

filter_valid_queries = RunnableLambda(valid_query_filter)

format_result = RunnableLambda(lambda x: {"query": x[1], "error": ""} if x[0] else {"query": "", "error": x[1]})

# chain which helps to pick or filter out the first correct query
pick_correct_query_chain = SessionRunnable(
    query_variables | query_prompt | model | validate_queries | format_result
).with_config(config={"tags": ["nl_to_sql_predict_gen_query_runnable"], "run_name": "Find Query"})
# chain which helps to filter out valid queries
filter_valid_queries_chain = SessionRunnable(query_variables | query_prompt | model | filter_valid_queries).with_config(
    config={"tags": ["nl_to_sql_predict_gen_query_runnable"], "run_name": "Find Query"}
)

chain = pick_correct_query_chain
