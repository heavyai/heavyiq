import asyncio

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

from heavyiq.langchain.utils import init_telemetrics

init_telemetrics()


async def main():
    from heavyiq.langgraph.graphs.sql_graph import graph

    inputs = {
        "question": "How many states are there?",
        "session_id": "Ny18ncWf6miTdy4Zw4DWRCe8is3cNZ6d",
        "allowed_tables": ["heavyai_us_counties"],
    }
    response = await graph.ainvoke(inputs, config={"configurable": {"session_id": inputs["session_id"]}})
    print(response)
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


if __name__ == "__main__":

    asyncio.run(main())
