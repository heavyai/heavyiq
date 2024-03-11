import asyncio
from typing import Coroutine

from fastapi.concurrency import run_in_threadpool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import HeavyDB

from ..heavydb.auto_sql_chain import chain as auto_sql_chain


class InputType(BaseModel):
    """Chain input type."""

    query_id: str | None = Field(default=None, description="Eval query id.")
    db_id: str = Field(..., description="HeavyDB database name.")
    question: str = Field(..., description="Natural Language question.")
    sql: str | None = Field(default=None, description="HeavyDB compatible SQL query.")
    enable_query_stats: bool = Field(default=True, description="Whether to calculate query stats or not.")


class OutputType(BaseModel):
    """Chain output type"""

    query_id: str | None = Field(default=None, description="Eval query id.")
    db_id: str = Field(..., description="HeavyDB database name.")
    gold_query: str = Field(..., description="Gold Query.")
    pred_query: str = Field(..., description="Predicted Query.")
    success: bool = Field(..., description="Boolean flag which indicates whether the prediction went correct or not.")
    status: str = Field(..., description="String which denotes the prediction status.")
    error: str = Field(default="", description="stores the error string.")
    query_stats: dict = Field(default={}, description="Query stats")


async def get_session_id(inputs: dict) -> str:
    """
    Create a db session against the input db_id using the credenials from the config.
    """
    return await HeavyDB.create_session_id_async(db_name=inputs["db_id"])


async def extract_gold_tables_from_query(inputs: dict) -> list[str]:
    """
    Helps to extract gold tables from the input SQL Query.
    """
    from heavyiq.cli.eval_commands.utils import extract_tables_from_query

    heavydb = await HeavyDB.from_session_async(session_id=inputs["session_id"])
    return extract_tables_from_query(heavydb._conn, inputs["sql"])


async def compare_tables(inputs: dict) -> dict:
    """
    Compares the extracted and predicted tables.
    """
    gold_tables, pred_tables = sorted(inputs["gold_tables"]), sorted(inputs["pred_tables"])
    error = ""
    if gold_tables != pred_tables:
        error = "TABLES_MISMATCH"
    return {**inputs, "error": error, "tables": pred_tables}


async def compare_and_format_output(inputs: dict) -> dict:
    """
    Formats the llm results according to the chain's output schema.
    """
    from heavyiq.cli.eval_commands.utils import sql_rate_reply

    sql_outputs = inputs["sql_outputs"]
    gold_tables, pred_tables, gold_query, pred_query, error = (
        sorted(inputs["gold_tables"]),
        sorted(sql_outputs["tables"]),
        inputs["sql"],
        sql_outputs["query"],
        sql_outputs["error"],
    )

    has_tables_mismatch = True if gold_tables != pred_tables else False

    # has error then pred query failed to pass the validation step
    # or the tables might get mismatched. In this case, don't calculate sql_rate_reply and query_stats
    # just return the function with necessary error details
    if error or has_tables_mismatch:
        status = "tables_mismatch" if has_tables_mismatch else "failed_to_generate_sql"
        return {
            "query_id": inputs["query_id"],
            "db_id": inputs["db_id"],
            "gold_query": gold_query,
            "pred_query": pred_query,  # failed sql
            "success": False,
            "status": status,
            "error": error,
            "query_stats": {},
        }

    db = await HeavyDB.from_session_async(session_id=inputs["session_id"])
    tasks = [sql_rate_reply(gold_query, pred_query, db=db)]
    if inputs["enable_query_stats"]:
        tasks.append(db.aquery_stats(pred_query))
    else:
        async_lambda: Coroutine = asyncio.coroutine(lambda x: {})  # type: ignore
        tasks.append(async_lambda)

    exception, query_stats, eval_res = None, {}, {"success": False, "status": "", "error": ""}
    try:
        eval_res, query_stats = await asyncio.gather(*tasks)
    except Exception as e:
        eval_res["success"] = False
        eval_res["status"] = "failed_to_calculate_query_stats"
        exception = str(e)

    # close the db connection
    db._conn._client.disconnect(db._conn.sessionid)
    db._conn.close()

    return {
        "query_id": inputs["query_id"],
        "db_id": inputs["db_id"],
        "gold_query": gold_query,
        "pred_query": pred_query,
        "success": eval_res["success"],
        "status": eval_res["status"],
        "error": exception or eval_res["error"],
        "query_stats": query_stats,
    }


get_session_id_lambda: Runnable = RunnableLambda(get_session_id)
extract_gold_tables: Runnable = RunnableLambda(extract_gold_tables_from_query)

chain: Runnable = (
    RunnablePassthrough().assign(session_id=get_session_id_lambda)
    | RunnablePassthrough().assign(gold_tables=extract_gold_tables)
    | RunnablePassthrough.assign(sql_outputs=auto_sql_chain)
    | RunnableLambda(compare_and_format_output)  # type: ignore
).with_types(
    input_type=InputType, output_type=OutputType  # type: ignore
)
