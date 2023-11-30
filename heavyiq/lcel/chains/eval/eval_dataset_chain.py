# Get tables list by passing db_id/db_name and question

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import HeavyDB

from ..heavydb.sql_chain import chain as sql_chain
from ..heavydb.table_chain import tables_list_chain


class InputType(BaseModel):
    """Chain input type."""

    question: str = Field(..., description="Natural Language question.")
    db_id: str = Field(..., description="HeavyDB database name.")
    query_id: str | None = Field(default=None, description="Eval query id.")


class OutputType(BaseModel):
    """Chain output type"""

    query_id: str | None = Field(default=None, description="Eval query id.")
    question: str = Field(..., description="Natural Language question.")
    db_name: str = Field(..., description="HeavyDB database name.")
    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list = Field(..., description="List of tables taken into consideration.")
    sql: str | None = Field(default=None, description="HeavyDB compatible SQL query.")
    sql_complexity: int | None = Field(default=None, description="SQL Query complexity.")
    sql_error: str | None = Field(default=None, description="Error string which might appears during SQL generation.")


async def get_session_id(inputs: dict) -> str:
    """
    Create a db session against the input db_id using the credenials from the config.
    """
    return await HeavyDB.create_session_id_async(db_name=inputs["db_id"])


async def format_output(inputs: dict) -> dict:
    """
    Formats the llm results according to the chain's output schema.
    """
    output = {
        "query_id": inputs.get("query_id"),
        "db_name": inputs["db_id"],
        "session_id": inputs["session_id"],
        "question": inputs["question"],
        "tables": inputs["tables"],
    }
    sql_output = inputs["sql_output"]
    # error = sql_output["error"]
    # if error:
    #     output.update({"sql": None, "sql_complexity": None, "sql_error": error})
    # else:
    output.update(
        {"sql": sql_output["query"], "sql_complexity": sql_output["sql_complexity"], "sql_error": sql_output["error"]}
    )
    return output


get_session_id_lambda: Runnable = RunnableLambda(get_session_id)
tables_runnable = RunnablePassthrough().assign(session_id=get_session_id_lambda) | tables_list_chain

chain: Runnable = (
    RunnablePassthrough().assign(session_id=get_session_id_lambda)
    | RunnablePassthrough().assign(tables=tables_list_chain)
    | RunnablePassthrough().assign(sql_output=sql_chain)
    | RunnableLambda(format_output)  # type: ignore
).with_types(
    input_type=InputType, output_type=OutputType  # type: ignore
)
