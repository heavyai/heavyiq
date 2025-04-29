import asyncio
import copy
import json
import operator
import os
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Any, Dict, Optional

import aiofiles
import vl_convert as vlc
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field, Json

from heavyiq.langchain.heavydb import HeavyDB, heavydb_context
from heavyiq.lcel.chains.heavydb.sql_chain import chain as sql_chain
from heavyiq.utils import add_limit_clause_to_query, extract_select_columns

from .logger_setup import FAILURE_EMOJI, START_EMOJI, SUCCESS_EMOJI, logger

# Model and prompts
# Define model and prompts we will use

chart_questions_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content="You are a dataset generation assistant helping train a chart-generation model."),
        HumanMessagePromptTemplate.from_template(
            """You are given the schemas of one or more related database tables:

{table_schema}

### Your Task:

Step 1: Generate **{n} natural language questions** that clearly request a **data visualization**, such as:
- "Create a bar chart showing total profit by region."
- "Visualize average quantity ordered per category."
- "Show a line chart of monthly sales trends per customer segment."
- "Generate a map chart showing total sales by state.
- "Plot a choropleth map of customer count by country."

Guidelines:
- The questions should reflect realistic **business intelligence use cases**.
- Use **one or more tables** where relevant — the model is expected to pick interconnected columns.
- Use appropriate **aggregations** (e.g., SUM, COUNT, AVERAGE) and **groupings** (e.g., by customer, by region, by month).
- Questions should cover a variety of **chart types**, including:
  - Bar charts
  - Line charts
  - Pie charts
  - Stacked/grouped bar charts
  - Histogram
  - Area charts
  - **Map charts** (e.g., choropleths or geospatial charts using location fields like country, state, region, coordinates)
- Keep questions **clear, chart-directed**, and **naturally phrased**.

Output Format:
Return a list of **chart_question** in JSON format.

### Example Output:

```json
[
  "Create a stacked bar chart comparing total sales by region and category.",
  "Show a line chart of total revenue per month for the top 5 customers.",
  "Visualize the average delivery delay by product sub-category.",
  "Generate a choropleth map showing number of orders by US state.",
  "Plot a map showing average order value by customer region."
]
```
Now generate {n} chart-style question using the given schemas.
"""
        ),
    ],
    input_variables=["table_schema", "n"],
)  # type: ignore

vega_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are a data visualization assistant that generates Vega-Lite specifications.\n\n"
                "You will be given:\n"
                "1. A chart-related natural language question.\n"
                "2. Schemas of the tables involved.\n"
                "3. A SQL query that fulfills the question.\n\n"
                "Your task is to generate a complete Vega-Lite JSON specification that visualizes the **result of the SQL query**. "
                "Make sure the chart type and encoding choices are appropriate for the query.\n\n"
                "The Vega-Lite spec should include:\n"
                '- `data`: use "url": "data.csv" (assume CSV result of the SQL query)\n'
                '- `mark`: choose based on the question (e.g., "bar", "line", "arc", etc.)\n'
                "- `encoding`: map X and Y axes based on group-by and aggregation logic\n"
                "- `title`: optional, based on the question\n\n"
                "**If the SQL result includes `latitude` and `longitude` fields:**\n"
                "- Create a layered Vega-Lite spec with a base map using `geoshape` (US counties from `https://vega.github.io/vega-datasets/data/us-10m.json`, format type `topojson`, feature `counties`).\n"
                "- Add a second layer with `mark: circle` to plot the points using `latitude` and `longitude` as `latitude` and `longitude` encodings.\n"
                '- Use `projection`: { "type": "albersUsa" } for both layers.\n\n'
                '- Ensure that circles are styled with **vibrant fill colors** (e.g., use a color scale with `"color": {"field": ..., "type": "quantitative", "scheme": "plasma"}` or a fixed color like `"#e45756"`).\n'
                "- The base map (`geoshape`) should use `fill: '#eee'` and `stroke: 'white'`.\n"
                "- The chart background should be white, but marks must remain clearly visible against it.\n\n"
                "Respond only with the Vega-Lite spec in JSON format."
            )
        ),
        HumanMessagePromptTemplate.from_template(
            template=(
                "**Question:**\n"
                "{question}\n\n"
                "**Table Schema:**\n"
                "{table_schema}\n\n"
                "**SQL Query:**\n"
                "{query}"
            )
        ),
    ],
    input_variables=["question", "table_schema", "query"],
)  # type: ignore


class ChartQuestions(BaseModel):
    questions: list[str]


class VegaLiteSpec(BaseModel):
    spec: Json[Any]


class VegaSpecError(Exception):
    pass


@lru_cache(maxsize=10)
def get_llm(model_choice: str) -> BaseChatModel:
    """
    Function to get llm instances by model name.
    ex: gemini-2.0-flash, gemini-2.5-flash-preview-04-17
    """
    logger.info(f"Using {model_choice} model.")
    if model_choice.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_choice, timeout=10, transport="rest")
    raise ValueError(f"Unknown model {model_choice}")


# This will be the overall state of the main graph.
class OverallState(BaseModel):
    database_name: str
    table_name: str
    image_folder: str  # folder path where vega spec images are being stored
    model_name: Optional[str] = "gemini-2.0-flash"
    table_schema: Optional[str] = ""
    input_viz_question: Optional[str] = ""
    input_sql: Optional[str] = ""
    db: Optional[HeavyDB] = None
    tables: list[str] = []
    n: int = Field(default=1, ge=0, description="Number of question pairs to generate.")

    questions: Annotated[list, operator.add]
    queries: Annotated[list, operator.add]
    vega_lite_specs: Annotated[list, operator.add]

    questions_with_sql: Annotated[list, operator.add]
    questions_with_error_sql: Annotated[list, operator.add]
    questions_with_vega: Annotated[list, operator.add]
    questions_with_vega_image: Annotated[list, operator.add]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# QuestionState usually gets passed to generate_sql node
class QuestionState(BaseModel):
    question: str
    db: HeavyDB
    tables: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VegaState(BaseModel):
    question: str
    query: str
    table_schema: str
    model_name: str


class VegaDataState(BaseModel):
    question: str
    query: str
    db: HeavyDB
    vega_spec: dict | Json
    image_folder: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


async def init_db(state: OverallState) -> dict:
    db = await HeavyDB.create_with_persistant_connection_async(db_name=state.database_name)
    return {"db": db, "tables": state.table_name.split(",")}


# This is the function we will use to generate the subjects of the jokes
async def fetch_table_schema(state: OverallState) -> dict:
    """
    Fetches table schema and updates the relevant state variable.
    """
    table_schema = await state.db.aget_table_info(
        table_names=state.tables,
        include_samples=False,
        include_top_k=True,
        include_timestamp=True,
        include_comments=True,
    )
    return {"table_schema": table_schema}


async def add_question(state: OverallState) -> dict[str, Any]:
    """
    Adds up the input question to the existing questions state.
    """
    return {"questions": [state.input_viz_question]}


async def generate_questions(state: OverallState) -> dict[str, Any]:
    """
    Invoke llm to generate n question pairs.
    """
    logger.info(f"{START_EMOJI} Generating viz questions...")
    llm = get_llm(state.model_name)
    chain = llm.with_structured_output(ChartQuestions)
    inputs = await chart_questions_prompt.aformat_messages(
        **{
            "table_schema": state.table_schema,
            "n": state.n,
        }
    )
    response: ChartQuestions = await asyncio.to_thread(chain.invoke, inputs)  # type: ignore
    logger.info(f"{SUCCESS_EMOJI} Generated viz questions.")
    return {"questions": response.questions}


async def generate_sql(state: QuestionState) -> dict[str, Any]:
    """
    Generates SQL for the given chart and SQL question.
    """
    logger.info(f"{START_EMOJI} Generating SQL using NL to SQL chain...")
    async with heavydb_context(state.db):
        sql_chain_response = await sql_chain.ainvoke(
            {"session_id": state.db._conn._session, "question": state.question, "tables": state.tables}
        )
        query = sql_chain_response["query"]
        if sql_chain_response["error"]:
            logger.error(f"{FAILURE_EMOJI} Generated SQL query failed validation.")
            return {"questions_with_error_sql": [(state.question, query)]}

    logger.info(f"{SUCCESS_EMOJI} Generated SQL query passes validation.")
    return {"queries": [query], "questions_with_sql": [(state.question, query)]}


generate_sql_retry_policy = RetryPolicy(max_attempts=2)


def continue_to_generate_vega(state: OverallState) -> list[Send]:
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    # Fan-out to generate_vega, ie. n number of generate_vega functions should be executed in parallel.
    return [
        Send(
            "generate_vega",
            VegaState(question=s[0], query=s[1], table_schema=state.table_schema, model_name=state.model_name),
        )
        for s in state.questions_with_sql
    ]


async def validate_vega_spec(spec: dict) -> bool:
    """
    Helps to validate the generated Vega-lite spec.
    """
    try:
        await asyncio.to_thread(vlc.vegalite_to_png, vl_spec=spec, vl_version="v5.15", scale=2)
    except Exception:
        return False
    else:
        return True


async def generate_vega(state: VegaState) -> dict[str, list[tuple[str, str, dict | str]]]:
    """
    Helps to generate vega-lite spec.
    """
    logger.info(f"{START_EMOJI} Generating Vega-lite spec...")
    llm = get_llm(state.model_name)
    chain = llm.with_structured_output(VegaLiteSpec)
    inputs = await vega_prompt.aformat_messages(
        **{
            "question": state.question,
            "table_schema": state.table_schema,
            "query": state.query,
        }
    )
    response: VegaLiteSpec = await asyncio.to_thread(chain.invoke, inputs)  # type: ignore
    is_valid = await validate_vega_spec(response.spec)
    if not is_valid:
        # raising error helps to retry this node which results in new spec generation
        logger.error(f"{FAILURE_EMOJI} Invalid Vega-lite spec.")
        raise VegaSpecError("Invalid Vega-lite spec")

    logger.info(f"{SUCCESS_EMOJI} Generated valid vega-lite spec.")
    return {"questions_with_vega": [(state.question, state.query, response.spec)]}


generate_vega_retry_policy = RetryPolicy(max_attempts=2)


def continue_to_generate_questions(state: OverallState) -> bool:
    """
    Whether continue to generate questions or not.
    """
    if not state.input_viz_question:
        return True
    return False


def continue_to_generate_single_sql_or_generate_vega(state: OverallState) -> Send:
    """
    Whether continue to generate single SQL query for vega spec.
    """
    if state.input_sql:
        # has input sq, so continue to generate vega
        return Send(
            "generate_vega",
            VegaState(
                question=state.input_viz_question,
                query=state.input_sql,
                table_schema=state.table_schema,
                model_name=state.model_name,
            ),
        )
    # ask to generate sql
    return Send("generate_sql", QuestionState(question=state.input_viz_question, db=state.db, tables=state.tables))


def continue_to_generate_sql(state: OverallState) -> list[Send]:
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send(
            "generate_sql",
            QuestionState(
                **{
                    "question": s,
                    "db": state.db,
                    "tables": state.tables,
                }
            ),
        )
        for s in state.questions
    ]


def continue_to_add_vega_data_and_generate_image(state: OverallState) -> list[Send]:
    """
    Return a list of Send to add vega data in parrallel.
    """
    return [
        Send(
            "add_vega_data_and_generate_image",
            VegaDataState(question=s[0], query=s[1], vega_spec=s[2], db=state.db, image_folder=state.image_folder),
        )
        for s in state.questions_with_vega
    ]


def to_iso(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


async def query_vega_data(query: str, db: HeavyDB, limit: int = 100) -> list[dict]:
    """
    Get vega data by querying the HeavyDB database.
    """
    query = add_limit_clause_to_query(query, limit=100)
    values = await db.arun(query, fetch="all", to_str=False)
    select_columns = extract_select_columns(query=query, remove_table_reference=True)
    if not values:
        return []
    # assert len(select_columns) == len(values[0])
    if len(select_columns) != len(values[0]):
        logger.error(f"{FAILURE_EMOJI} Invalid select columns!")
        logger.info(query, len(select_columns), len(values[0]))
        logger.info(select_columns)
        logger.info(values[0])
        return []
    formatted_values = [dict(zip(select_columns, [to_iso(v) for v in row])) for row in values]
    return formatted_values


async def save_vega_png(spec: dict, image_folder: str) -> str:
    """
    Save vega spec to png.
    """
    filepath = os.path.join(image_folder, f"{uuid.uuid4().hex[:8]}.png")
    try:
        png_bytes = await asyncio.to_thread(vlc.vegalite_to_png, vl_spec=spec, vl_version="v5.15", scale=2)
    except Exception as e:
        logger.error(f"{FAILURE_EMOJI} Failed to save vega image, {e}")
        return ""
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(png_bytes)
    return filepath


VEGA_TO_IMAGE_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_png_worker.py")


async def save_vega_png_subprocess(spec: dict, image_folder: str, timeout: int = 10) -> str:
    """
    Runs vl_convert.vegalite_to_png in an isolated subprocess, so that rust errors can be easily catched and skiped.
    """
    filename = f"{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(image_folder, filename)

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            VEGA_TO_IMAGE_WORKER_PATH,  # Path to the helper script
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        input_data = json.dumps(spec).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_data), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error(f"{FAILURE_EMOJI} Rendering timed out after {timeout}s")
            return ""

        if proc.returncode != 0:
            logger.error(f"{FAILURE_EMOJI} Subprocess failed: {stderr.decode()}")
            return ""

        async with aiofiles.open(filepath, "wb") as f:
            await f.write(stdout)

        return filepath

    except Exception as e:
        logger.error(f"{FAILURE_EMOJI} Failed to save vega image: {e}")
        return ""


async def add_vega_data_and_generate_image(state: VegaDataState):
    """
    Query and add vega data and then generate image.
    """
    logger.info(f"{START_EMOJI} Querying vega data...")
    values = await query_vega_data(state.query, state.db)
    spec = copy.deepcopy(state.vega_spec)
    spec["data"] = {"values": values}
    image_path = await save_vega_png_subprocess(spec, state.image_folder)
    if image_path:
        logger.info(f"{SUCCESS_EMOJI} Chart image generated successfully.")
    else:
        logger.error(f"{FAILURE_EMOJI} Chart image failed to generate.")
    # make sure the spec contain not more than 3 values for writing into output csv
    spec["data"] = {"values": values[:3]}
    return {"questions_with_vega_image": [(state.question, state.query, spec, image_path)]}


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("init_db", init_db)
graph.add_node("fetch_table_schema", fetch_table_schema)
graph.add_node("generate_questions", generate_questions)
graph.add_node("add_question", add_question)
graph.add_node("generate_sql", generate_sql)
graph.add_node("generate_vega", generate_vega, retry=generate_vega_retry_policy)
graph.add_node("add_vega_data_and_generate_image", add_vega_data_and_generate_image)

graph.add_edge(START, "init_db")
graph.add_edge("init_db", "fetch_table_schema")
graph.add_conditional_edges(
    "fetch_table_schema", continue_to_generate_questions, {True: "generate_questions", False: "add_question"}
)
# False
graph.add_conditional_edges("add_question", continue_to_generate_single_sql_or_generate_vega)

# True
graph.add_conditional_edges("generate_questions", continue_to_generate_sql, ["generate_sql"])  # type: ignore
graph.add_conditional_edges("generate_sql", continue_to_generate_vega, ["generate_vega"])  # type: ignore
graph.add_conditional_edges(
    "generate_vega", continue_to_add_vega_data_and_generate_image, ["add_vega_data_and_generate_image"]  # type: ignore
)
graph.add_edge("add_vega_data_and_generate_image", END)
app = graph.compile()
