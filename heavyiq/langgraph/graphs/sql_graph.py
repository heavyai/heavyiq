import operator
from typing import Annotated, TypedDict

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda
from langgraph.channels.context import Context
from langgraph.graph import END, START, StateGraph

from heavyiq.langchain.heavydb import get_config
from heavyiq.langgraph.utils import HeavyDBContext, make_heavydb_context
from heavyiq.lcel.chains.heavydb.relevant_info_chain import chain as relevant_info_chain
from heavyiq.lcel.chains.heavydb.sql_chain import chain as sql_chain
from heavyiq.lcel.chains.heavydb.table_chain import tables_dict_chain

CONFIG = get_config()


class OverallState(BaseModel):
    """
    OverallState for Natural Language question to SQL graph.
    """

    question: str
    session_id: str
    tables: Annotated[list[str], operator.add]
    allowed_tables: Annotated[list[str], operator.add]
    """
    context gets called and assigned before the graph starts
    and it won't be appear in graph response or while make persistant state
    """
    context: Annotated[HeavyDBContext, Context(make_heavydb_context)]
    sql: str | None
    answer: str | None
    snippet_ids: Annotated[list[str], operator.add]
    relevant_info: str | None


class ConfigSchema(TypedDict):
    """
    List of config items passed to the state graph upon invoking.
    """

    session_id: str


# Define the function that determines whether to continue or not
def should_continue_after_tables(state: OverallState) -> bool:
    """
    Whether to continue or not after finding the tables.
    """
    if state.tables:
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


# Define a new graph
workflow = StateGraph(OverallState, config_schema=ConfigSchema)


##### Nodes

workflow.add_node("rag_retrieve_relevant_info", relevant_info_chain)
workflow.add_node("dont_retrieve_relevant_info", lambda x: {"relevant_info": ""})
workflow.add_node("merge_state_after_snippets", lambda x: {"question": x.question})

workflow.add_node(
    "retrieve_tables",
    RunnableLambda(lambda x: x.dict()) | tables_dict_chain,
)
workflow.add_node(
    "generate_sql", RunnableLambda(lambda x: x.dict()) | sql_chain | RunnableLambda(lambda x: {"sql": x.get("query")})
)
#####

# Edges
# Step 1: Decides whether to route to retrieve relevant_info runnable or not
workflow.add_conditional_edges(
    START, route_to_retrieve_relevant_info, {True: "rag_retrieve_relevant_info", False: "dont_retrieve_relevant_info"}
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
        False: "generate_sql",
    },
)

# # End if there wasn't any tables found
workflow.add_conditional_edges(
    "retrieve_tables",
    should_continue_after_tables,
    {
        True: "generate_sql",
        False: END,
    },
)

workflow.add_edge("generate_sql", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
