import asyncio
import operator
from typing import Annotated, Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field, Json

from heavyiq.langchain.heavydb import HeavyDB, heavydb_context
from heavyiq.lcel.chains.heavydb.sql_chain import chain as sql_chain

# Model and prompts
# Define model and prompts we will use

chart_questions_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content="You are a dataset generation assistant helping train a chart-generation model."),
        HumanMessagePromptTemplate.from_template(
            """Given the following table schema:

{table_schema}

### Your Task:

Step 1: Generate **{n} natural language questions** that clearly request a **data visualization**, such as:
- "Create a bar chart showing total profit by region."
- "Visualize average quantity per category."
- "Show a line chart of sales trends over months."

Guidelines:
- Focus on **aggregations** (e.g., sum, average, count) and **groupings** (e.g., by region, category, month).
- Keep questions natural but **chart-directed**.
- Use business intelligence use cases.

Output Format:
Return a list of **chart_question** in JSON format.

### Example Output:

```json
[
  "Create a bar chart showing total profit by region.",
  "Visualize the average quantity ordered per category."
]
```
Now generate {n} chart-style question.
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
                "2. The schema of the table involved.\n"
                "3. A SQL query that fulfills the question.\n\n"
                "Your task is to generate a complete Vega-Lite JSON specification that visualizes the **result of the SQL query**. "
                "Make sure the chart type and encoding choices are appropriate for the query.\n\n"
                "The Vega-Lite spec should include:\n"
                '- `data`: use "url": "data.csv" (assume CSV result of the SQL query)\n'
                '- `mark`: choose based on the question (e.g., "bar", "line", "arc", etc.)\n'
                "- `encoding`: map X and Y axes based on group-by and aggregation logic\n"
                "- `title`: optional, based on the question\n\n"
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


# model = ChatOpenAI(model="o3-mini")


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", timeout=10, transport="rest")


# This will be the overall state of the main graph.
class OverallState(BaseModel):
    database_name: str
    table_name: str
    table_schema: Optional[str] = ""
    db: Optional[HeavyDB] = None
    n: int = Field(default=10, ge=0, description="Number of question pairs to generate.")

    questions: Annotated[list, operator.add]
    queries: Annotated[list, operator.add]
    vega_lite_specs: Annotated[list, operator.add]

    questions_with_sql: Annotated[list, operator.add]
    questions_with_vega: Annotated[list, operator.add]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# QuestionState usually gets passed to generate_sql node
class QuestionState(BaseModel):
    question: str
    db: HeavyDB
    table: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VegaState(BaseModel):
    question: str
    query: str
    table_schema: str


async def init_db(state: OverallState) -> dict:
    db = await HeavyDB.create_with_persistant_connection_async(db_name=state.database_name)
    return {"db": db}


# This is the function we will use to generate the subjects of the jokes
async def fetch_table_schema(state: OverallState) -> dict:
    """
    Fetches table schema and updates the relevant state variable.
    """
    table_schema = await state.db.aget_table_info(
        table_names=[state.table_name],
        include_samples=False,
        include_top_k=True,
        include_timestamp=True,
        include_comments=True,
    )
    return {"table_schema": table_schema}


async def generate_questions(state: OverallState):
    """
    Invoke llm to generate n question pairs.
    """
    chain = llm.with_structured_output(ChartQuestions)
    inputs = await chart_questions_prompt.aformat_messages(
        **{
            "table_schema": state.table_schema,
            "n": state.n,
        }
    )
    response: ChartQuestions = await asyncio.to_thread(chain.invoke, inputs)  # type: ignore
    return {"questions": response.questions}


async def generate_sql(state: QuestionState):
    """
    Generates SQL for the given chart and SQL question.
    """
    async with heavydb_context(state.db):
        sql_chain_response = await sql_chain.ainvoke(
            {"session_id": state.db._conn._session, "question": state.question, "tables": [state.table]}
        )
        query = sql_chain_response["query"]
    return {"queries": [query], "questions_with_sql": [(state.question, query)]}


def continue_to_generate_vega(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    # Fan-out to generate_vega, ie. n number of generate_vega functions should be executed in parallel.
    return [
        Send(
            "generate_vega",
            VegaState(**{"question": s[0], "query": s[1], "table_schema": state.table_schema}),
        )
        for s in state.questions_with_sql
    ]


async def generate_vega(state: VegaState) -> dict[str, list[tuple[str, str, dict | str]]]:
    chain = llm.with_structured_output(VegaLiteSpec)
    inputs = await vega_prompt.aformat_messages(
        **{
            "question": state.question,
            "table_schema": state.table_schema,
            "query": state.query,
        }
    )
    response: VegaLiteSpec = await asyncio.to_thread(chain.invoke, inputs)  # type: ignore

    return {"questions_with_vega": [(state.question, state.query, response.spec)]}


def continue_to_generate_sql(state: OverallState):
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
                    "table": state.table_name,
                }
            ),
        )
        for s in state.questions
    ]


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("init_db", init_db)
graph.add_node("fetch_table_schema", fetch_table_schema)
graph.add_node("generate_questions", generate_questions)
graph.add_node("generate_sql", generate_sql)
graph.add_node("generate_vega", generate_vega)

graph.add_edge(START, "init_db")
graph.add_edge("init_db", "fetch_table_schema")
graph.add_edge("fetch_table_schema", "generate_questions")
graph.add_conditional_edges("generate_questions", continue_to_generate_sql, ["generate_sql"])
graph.add_conditional_edges("generate_sql", continue_to_generate_vega, ["generate_vega"])
graph.add_edge("generate_vega", END)
app = graph.compile()
