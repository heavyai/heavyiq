import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Sequence, cast

from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field, Json

from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import get_groq_chat_llm
from heavyiq.langchain.utils import aget_table_info_from_cache_or_calculate
from heavyiq.lcel.chains.heavydb.answer_chain import derive_column_metadata_from_sql
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_vega_lite_prompt_runnable, vega_chat_prompt, vega_lite_chat_pormpt


class ChartChainInputType(BaseModel):
    """
    ChartChain Input type.
    """

    question: str = Field(..., description="Natural Language question relevant to charting.")
    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list = Field(..., description="List of tables to consider.")
    query: str = Field(..., description="SQL Query")

    class Config:
        arbitrary_types_allowed = True


class ChartChainOutputType(BaseModel):
    """
    ChartChain Output type.
    """

    vega_lite_spec: Json = Field(..., description="Generated Vega Lite Spec")

    class Config:
        arbitrary_types_allowed = True


def data_placeholder(_: dict) -> str:
    return '"data": {"name": "table"}'


def apply_limit_to_query(query: str, limit: int = 5) -> str:
    """
    Transform a normal select query into a limit query.
    """
    # Normalize whitespace and remove trailing semicolon
    query = query.strip().rstrip(";")

    # Check if the query already has a LIMIT clause
    limit_pattern = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
    if limit_pattern.search(query):
        query = limit_pattern.sub(f"LIMIT {limit}", query)
    else:
        query += f" LIMIT {limit}"

    return query + ";"


async def run_sql_query(kwargs: dict) -> str:
    """
    Run the sql query against heavydb and return the resultset in string format.
    """
    session_id, query = kwargs.get("session_id"), kwargs.get("query")
    heavydb = await get_db(session_id)
    limit_query = apply_limit_to_query(query=query, limit=5)
    sql_result = await heavydb.arun(limit_query, to_str=True)
    return cast(str, sql_result)


async def derive_projected_columns(kwargs: dict) -> dict[str, list[str]]:
    """
    Derive only the projected columns from the input query.
    """
    db = await get_db(kwargs["session_id"])
    query = kwargs.get("sql_query") or kwargs.get("query")
    columns_mapping = await db.aretrieve_columns_from_query(query)
    # projected columns in the SQL query which are essential to define it in the
    # prompt along with their type
    projected_columns: dict = defaultdict(list)
    # ex: {"car": ["ID", "model_name", "price"]}, here car is the table and ID, model_name, price are the projected columns
    for detail in columns_mapping.values():
        _, table, column = detail
        if column not in projected_columns[table]:
            projected_columns[table].append(column)
    return projected_columns


async def fetch_schemas_for_tables(kwargs: dict) -> str:
    """
    Fetch schemas for the given tables.
    """

    task_1 = aget_table_info_from_cache_or_calculate(
        session=kwargs.get("session_id"),
        tables=kwargs.get("tables"),
        include_samples=False,
        include_top_k=False,
        include_timestamp=False,
        include_comments=False,
    )
    task_2 = derive_projected_columns(kwargs)
    table_schema, projected_columns = await asyncio.gather(task_1, task_2)
    splitted_schemas = table_schema.split("\n\n")

    schemas_dict = {}  # contain table_name and raw schema mapping
    projected_columns_with_detail = defaultdict(list)
    for i in splitted_schemas:
        table_name = i.strip().split("(", 1)[0].strip().split()[-1]
        schemas_dict[table_name] = i

    for table, cols in projected_columns.items():
        for col in cols:
            match = re.search(r"(?m)^(" + col + r".*?)(?:\);)?$", schemas_dict[table])
            if match:
                projected_columns_with_detail[table].append(match.group(1))

    schemas_with_projected_columns = ""
    for table, detail_projected_cols in projected_columns_with_detail.items():
        if detail_projected_cols:
            schemas_with_projected_columns += f"CREATE TABLE {table} (\n"
            schemas_with_projected_columns += "\n".join(detail_projected_cols)
            schemas_with_projected_columns += ");\n\n"

    return schemas_with_projected_columns.strip()


def output_formatter(values: str | AIMessage) -> dict:
    content = values.content if isinstance(values, AIMessage) else values
    if isinstance(content, str) and "```json" in content:
        content = content.strip("```json").strip("```")
    return {"vega_lite_spec": content}


format_output_rbl = RunnableLambda(output_formatter)

# Default prompt and llm
prompt_rbl = to_vega_lite_prompt_runnable
llm_rbl = llm_runnable.with_config(configurable={"llm": "default_llm"}).bind(extra_body={"guided_regex": "{.*}"})
# Groq prompt and llm
# prompt_rbl = vega_lite_chat_pormpt
# llm_rbl = get_groq_chat_llm()

prompt_variables = RunnablePassthrough.assign(
    data_placeholder=data_placeholder,
    table_schema=RunnableLambda(fetch_schemas_for_tables),
    sql_query=lambda k: k["query"],
    sample_data=RunnableLambda(run_sql_query),
    question=lambda y: y["question"],
)

chain = (
    (prompt_variables | prompt_rbl | llm_rbl | format_output_rbl)
    .with_types(input_type=ChartChainInputType, output_type=ChartChainOutputType)  # type: ignore
    .with_config(
        config={"tags": ["NLtoVegaLiteSpecChainRunnable"], "run_name": "NL to Vega Lite Spec Chain"}  # type: ignore
    )
)
