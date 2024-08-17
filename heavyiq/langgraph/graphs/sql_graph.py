import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langgraph.channels.context import Context
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from heavyiq.langchain.heavydb import get_config
from heavyiq.langgraph.utils import HeavyDBContext, make_heavydb_context, remove_duplicates
from heavyiq.lcel.chains.heavydb.answer_chain import chain as answer_chain
from heavyiq.lcel.chains.heavydb.relevant_info_chain import chain as relevant_info_chain
from heavyiq.lcel.chains.heavydb.sql_chain import get_table_info, query_llm, query_prompt, validate_and_revise_chain
from heavyiq.lcel.chains.heavydb.table_chain import tables_dict_chain

CONFIG = get_config()


class OverallState(BaseModel):
    """
    OverallState for Natural Language question to SQL graph.
    """

    question: str
    session_id: str
    tables: Annotated[list[str], remove_duplicates]
    allowed_tables: Annotated[list[str], remove_duplicates]
    """
    context gets called and assigned before the graph starts
    and it won't be appear in graph response or while make persistant state
    """
    context: Annotated[HeavyDBContext | None, Context(make_heavydb_context)]
    query: str | None
    answer: str | None
    snippet_ids: Annotated[list[str], remove_duplicates]
    relevant_info: str | None
    should_continue: bool | None = False
    table_info: str | None
    sql_complexity: int | None
    error: str | None
    results: str | None


class ConfigSchema(TypedDict):
    """
    List of config items passed to the state graph upon invoking.
    """

    session_id: str
    do_generate_answer: bool


# runnable which helps to convert the pydantic model object to dict
# which then passed to the following runnable as input
model_to_dict_runnable: Runnable = RunnableLambda(lambda x: x.dict())


# Define the function that determines whether to continue or not
def should_continue_after_tables(state: OverallState) -> bool:
    """
    Whether to continue or not after finding the tables.
    """
    if state.tables:
        return True
    return False


def should_continue_after_sql_generation(state: OverallState) -> bool:
    """
    Whether to continue after sql generation and validation.
    """
    return state.should_continue


def should_generate_answer(state: OverallState, config: RunnableConfig) -> bool:
    """
    Decides whether to generate answer from sql or not.
    """
    if config["configurable"].get("do_generate_answer", False) == True:
        return True
    return False


def route_to_retrieve_relevant_info(state: OverallState) -> bool:
    """
    Route to get relevant_info or not.
    """
    if CONFIG.enable_rag and (
        CONFIG.custom_prompt_nl_to_tables_include_relevant_info or CONFIG.custom_prompt_nl_to_sql_include_relevant_info
    ):
        return True
    return False


def route_to_retrieve_tables(state: OverallState) -> bool:
    """
    Route to retrieve tables or not.
    """
    if state.tables:
        return False
    return True


def prepare_table_chain_inputs(state: OverallState):
    """
    Prepare the inputs for table chain runnable, especially converting pydantic state to dict.
    """
    {
        "question": state.question,
        "session_id": state.session_id,
        "allowed_tables": state.allowed_tables,
        "snippet_ids": state.snippet_ids,
        "relevant_info": state.relevant_info,
    }


def append_relevant_info_to_table_info_if_not_exists(overallstate: OverallState) -> dict:
    """
    Appends relevevant_info to table_info text.
    """
    state = overallstate.dict()
    table_info, relevant_info = state["table_info"], state["relevant_info"]
    if relevant_info and "Relevant info:" not in table_info:
        table_info += "\n" + relevant_info
    return {"table_info": table_info, "input": state["question"], "relevant_info": state["relevant_info"]}


# Define a new graph
workflow = StateGraph(OverallState, config_schema=ConfigSchema)
memory = AsyncSqliteSaver.from_conn_string(":memory:")

# Nodes
workflow.add_node("start", lambda x: {"should_continue": False})
workflow.add_node("rag_retrieve_relevant_info", model_to_dict_runnable | relevant_info_chain)
workflow.add_node("dont_retrieve_relevant_info", lambda x: {"relevant_info": ""})
workflow.add_node("merge_state_after_snippets", lambda x: {"question": x.question})

workflow.add_node(
    "retrieve_tables",
    model_to_dict_runnable | tables_dict_chain,
)

workflow.add_node("get_table_info", {"table_info": model_to_dict_runnable | get_table_info})  # this adds relevant_info

# Sub Graph
query_sub_graph = StateGraph(OverallState)
query_sub_graph.add_node(
    "generate_query",
    {
        "query": (
            RunnableLambda(
                append_relevant_info_to_table_info_if_not_exists
            )  # retrun all state values so that the prompt runnable will get all
            | query_prompt
            | query_llm
            | StrOutputParser()
        )
    },
)


def final_node(state: OverallState) -> dict:
    """
    Final node.
    """
    do_include = ["answer", "tables", "query", "error", "results", "sql_complexity", "snippet_ids"]
    return {k: v for k, v in state.dict().items() if k in do_include}


query_sub_graph.add_node(
    "validate_and_revise_query",
    RunnableLambda(
        lambda x: {"sql_cmd": x.query, "max_revisions": CONFIG.max_retries_nl_to_sql, "session_id": x.session_id}
    )
    | validate_and_revise_chain,
)
# Sub-graph edges
query_sub_graph.add_edge(START, "generate_query")
query_sub_graph.add_edge("generate_query", "validate_and_revise_query")
query_sub_graph.add_edge("validate_and_revise_query", END)


workflow.add_node("generate_sql", query_sub_graph.compile())
workflow.add_node("generate_answer", model_to_dict_runnable | answer_chain)
workflow.add_node("final", final_node)

# Edges
# Step 1: Decides whether to route to retrieve relevant_info runnable or not
workflow.add_edge(START, "start")
workflow.add_conditional_edges(
    "start", route_to_retrieve_relevant_info, {True: "rag_retrieve_relevant_info", False: "dont_retrieve_relevant_info"}
)
workflow.add_edge("rag_retrieve_relevant_info", "merge_state_after_snippets")
workflow.add_edge("dont_retrieve_relevant_info", "merge_state_after_snippets")

# # Step 2
# # We now add a conditional edge to find whether we continue with
# # finding tables or not
workflow.add_conditional_edges(
    "merge_state_after_snippets",
    route_to_retrieve_tables,
    {
        True: "retrieve_tables",
        False: "get_table_info",
    },
)

# # End if there wasn't any tables found
workflow.add_conditional_edges(
    "retrieve_tables",
    should_continue_after_tables,
    {
        True: "get_table_info",
        False: END,
    },
)
workflow.add_edge("get_table_info", "generate_sql")

workflow.add_conditional_edges("generate_sql", should_generate_answer, {True: "generate_answer", False: "final"})
workflow.add_edge("generate_answer", "final")
workflow.add_conditional_edges("final", should_continue_after_sql_generation, {True: "generate_sql", False: END})

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile(checkpointer=memory, interrupt_after=["final"])
