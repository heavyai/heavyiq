# Get tables list by passing db_id/db_name and question

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import HeavyDB

from ..heavydb.table_chain import tables_list_chain


class InputType(BaseModel):
    """Chain input type."""

    query_id: str | None = Field(default=None, description="Eval query id.")
    db_id: str = Field(..., description="HeavyDB database name.")
    question: str = Field(..., description="Natural Language question.")
    sql: str | None = Field(default=None, description="HeavyDB compatible SQL query.")


class OutputType(BaseModel):
    """Chain output type"""

    query_id: str | None = Field(default=None, description="Eval query id.")
    db_id: str = Field(..., description="HeavyDB database name.")
    gold_tables: list[str] = Field(..., description="List of gold tables.")
    pred_tables: list[str] = Field(..., description="List of predicted tables.")
    success: bool = Field(..., description="Boolean flag which indicates whether the prediction went correct or not.")
    status: str = Field(..., description="String which denotes the prediction status.")
    error: str = Field(default="", description="stores the error string.")


async def get_session_id(inputs: dict) -> str:
    """
    Create a db session against the input db_id using the credenials from the config.
    """
    return await HeavyDB.create_session_id_async(db_name=inputs["db_id"])


async def compare_and_format_output(inputs: dict) -> dict:
    """
    Formats the llm results according to the chain's output schema.
    """
    gold_tables, pred_tables = sorted(inputs["gold_tables"]), sorted(inputs["pred_tables"])
    output = {
        "query_id": inputs.get("query_id"),
        "db_id": inputs["db_id"],
        "gold_tables": gold_tables,
        "pred_tables": pred_tables,
    }
    if gold_tables == pred_tables:
        output.update(
            {
                "success": True,
                "status": "success",
                "error": "",
            }
        )
    else:
        output.update(
            {
                "success": False,
                "status": "TABLES_MISMATCH",
                "error": "",
            }
        )

    return output


async def extract_gold_tables_from_query(inputs: dict) -> list[str]:
    """
    Helps to extract gold tables from the input SQL Query.
    """
    from heavyiq.cli.eval_commands.utils import extract_tables_from_query

    heavydb = await HeavyDB.from_session_async(session_id=inputs["session_id"])
    return extract_tables_from_query(heavydb._conn, inputs["sql"])


get_session_id_lambda: Runnable = RunnableLambda(get_session_id)
extract_gold_tables: Runnable = RunnableLambda(extract_gold_tables_from_query)


chain: Runnable = (
    RunnablePassthrough().assign(session_id=get_session_id_lambda)
    | RunnablePassthrough.assign(gold_tables=extract_gold_tables, pred_tables=tables_list_chain)
    | RunnableLambda(compare_and_format_output)  # type: ignore
).with_types(
    input_type=InputType, output_type=OutputType  # type: ignore
)
