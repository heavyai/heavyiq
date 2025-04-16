import asyncio
import getpass
import os

from dotenv import load_dotenv


async def main():
    from heavyiq.langchain.utils import init_telemetrics

    from .chart_graph import app

    # graph = app.get_graph()
    # graph.draw_png("chat_graph.png")
    # graph.print_ascii()
    # init_telemetrics()
    # # # Call the graph: here we call it to generate a list of jokes
    # inputs = {"database_name": "ca_tweets", "table_name": "ca_tweets", "n": 10}
    # async for s in app.astream(inputs):
    #     print(s)


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    asyncio.run(main())
