import asyncio
import itertools
import logging
import sys

from langchain_core.messages import SystemMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.prebuilt import create_react_agent

logging.getLogger().setLevel(logging.ERROR)  # hide warning log

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"


def print_answer(final_values: dict) -> None:
    has_query_error = True if final_values["error"] else False
    # print(f"Has Query Error? {MAGENTA}{has_query_error}{RESET}")
    if has_query_error:
        print(f'{RED}{final_values["error"]}{RESET}')
    else:
        print(f'{GREEN}{final_values["query"]}{RESET}')
        answer, results = final_values.get("answer"), final_values.get("results")
        if results:
            print(f"{BLUE}{results}{RESET}")
        if answer:
            print(f"{MAGENTA}{answer}{RESET}")


done = False


async def spinner():
    for char in itertools.cycle(["|", "/", "-", "\\"]):
        if done:
            break
        sys.stdout.write(f"\r{char} ")
        sys.stdout.flush()
        await asyncio.sleep(0.1)
    sys.stdout.write("\r  \r")


async def main():
    """
    Program start.
    """
    from heavyiq.langchain.heavydb import HeavyDB
    from heavyiq.langchain.utils import init_telemetrics
    from heavyiq.langgraph.graphs.sql_graph import graph

    init_telemetrics()
    global done
    dbname = input(f"{MAGENTA}Database Name: {RESET}")
    allowed_tables = input(f"{GREEN}Allowed Tables: {RESET}")
    session_id = await HeavyDB.create_session_id_async(db_name=dbname)
    question = input(f"{YELLOW}Question: {RESET}")
    task = asyncio.create_task(spinner())
    thread_id = 5
    inputs = {
        "question": question,
        "session_id": session_id,
        "allowed_tables": allowed_tables.split(","),
    }
    config = {"configurable": {"session_id": inputs["session_id"], "thread_id": thread_id, "do_generate_answer": False}}

    try:
        final_values = None
        async for event in graph.astream_events(inputs, config=config, version="v2"):
            event_name, ee, event_data = event["name"], event["event"], event["data"]
            final_output = event_data.get("output", None)
            if final_output:
                # print the event_name only when the runnable produces the output
                # print(event_name, ee)
                final_values = final_output
        done = True
        await task
        print_answer(final_values)
        while True:
            new_question = input(f"{YELLOW}Question (q): {RESET}")
            if new_question.lower() == "q":
                print(f"{BLUE}Operation cancelled by user.{RESET}")
                break
            done = False
            task = asyncio.create_task(spinner())
            # reset all
            await graph.aupdate_state(
                config,
                {
                    "question": new_question,
                    "should_continue": True,
                    "session_id": inputs["session_id"],
                    "snippet_ids": [],
                    "relevant_info": None,  # reset relevant_info for next question
                    "sql_complexity": None,
                    "error": None,
                    "query": None,
                    "answer": None,
                    "results": None,
                },
                # as_node="generate_sql",
            )
            # print("---\n---\nUpdated state!")
            # updated_state = await graph.aget_state(config)
            # print(updated_state.values)
            # print("=" * 100)
            async for event in graph.astream_events(None, config, version="v2"):
                event_name, ee, event_data = event["name"], event["event"], event["data"]
                final_output = event_data.get("output", None)
                if final_output:
                    # print the event_name only when the runnable produces the output
                    # print(event_name, ee)
                    final_values = final_output

            done = True
            await task
            print_answer(final_values)

        # Draw image
        graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path="/Users/avinash/Heavy/heavynl/graph.png",
            # draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
            draw_method=MermaidDrawMethod.API,
        )
        # close the sqlite db connection
    except Exception as e:
        raise e
    finally:
        done = True
        await graph.checkpointer.conn.close()
        wait_for_all_tracers()
        await task


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


async def foo():
    from heavyiq.langchain.llms import LLMType, get_llm_by_type
    from heavyiq.lcel.chains.heavydb.sql_chain import nl_to_sql_llm_rbl
    from heavyiq.lcel.chains.utils import get_value_from_runnable_binding

    system_message = 'You are an expert data analyst adept at writing SQL queries to answer user questions.\nYou have access to the following relational tables, with schemas below.\n\nAlongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.\n\nCREATE TABLE heavyai_us_counties (\nid TEXT\nfips TEXT (02201, 15009, 26115 ...)\ncounty TEXT (Washington County, Jefferson County, Franklin County ...)\nstate TEXT (TX, GA, VA ...)\nname TEXT (Prince of Wales-Outer Ketchikan Census Area, AK, Maui County, HI, Monroe County, MI ...)\ngeom MULTIPOLYGON);\n\nWrite a SQL query to answer the following question:\n'
    # This could also be a SystemMessage object
    # system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")
    # model = get_value_from_runnable_binding(nl_to_sql_llm_rbl)
    model = get_llm_by_type(LLMType.NL_TO_SQL_CHAT)
    app = create_react_agent(model, tools, state_modifier=system_message)

    query = "Find the names of counties in states that have more than 50 counties."

    messages = app.invoke({"messages": [("user", query)]})
    print(
        {
            "input": query,
            "output": messages["messages"][-1].content,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
