# SQL Generations chain
# chain used to generate n SQL queries by turning off the beam search
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


async def sql_list_validator(sql_queries: Sequence[str]) -> tuple[bool, str]:
    """
    Validates list of SQL queries and picks the right one.
    """
    heavydb = heavydb_var.get()

    async def validate(query: str) -> tuple[bool, str]:
        print(query)
        try:
            await heavydb.avalidate_query(query)
        except Exception as e:
            print(e)
            print("=" * 100)
            return False, extract_error_message_from_exception(e)
        else:
            return True, ""

    error = ""
    for query in sql_queries:
        success, error = await validate(query)
        if success:
            return True, query

    return False, error


validate_queries = RunnableLambda(sql_list_validator).with_config(
    config={
        "run_name": "SQL Query Validation",
        "tags": ["intermediate-step"],
        "metadata": {"step": "Validating SQL query."},
    }
)

format_result = RunnableLambda(lambda x: {"query": x[1], "error": ""} if x[0] else {"query": "", "error": x[1]})

query_runnable = SessionRunnable(query_variables | query_prompt | model | validate_queries | format_result).with_config(
    config={"tags": ["nl_to_sql_predict_query_runnable"], "run_name": "Find Query"}
)  # type: ignore
