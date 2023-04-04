import click
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.langchain.logging import log_agent_call
from modules.langchain.heavydb import HeavyDB
from modules.langchain.agents.agent_toolkits.heavydb.base import create_heavydb_agent
from modules.langchain.agents.agent_toolkits.heavydb.toolkit import HeavyDBToolkit

load_dotenv()


@click.command()
@click.option("--model", default="text-davinci-003", help="Specify the model to use for the LLM.")
@click.option("--temperature", default=0.0, type=float, help="What sampling temperature to use.")
@click.option(
    "--max-tokens",
    default=256,
    type=int,
    help="The maximum number of tokens to generate in the completion. -1 means to use model limit. (Chat: N/A)",
)
@click.option("--n", default=1, type=int, help="How many completions to generate for each prompt.")
@click.option(
    "--best-of",
    default=1,
    type=int,
    help="Generates best_of completions server-side and returns the 'best'. (Chat: N/A)",
)
@click.option(
    "--batch-size",
    default=20,
    type=int,
    help="Batch size to use when passing multiple documents to generate. (Chat: N/A)",
)
@click.option("--max-retries", default=6, type=int, help="Maximum number of retries to make when generating.")
@click.argument("question")
def main(
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    n: int,
    best_of: int,
    batch_size: int,
    max_retries: int,
):
    llm: ChatOpenAI | OpenAI
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        llm = ChatOpenAI(model_name=model, client=None, n=n, max_retries=max_retries, temperature=temperature)
    else:
        llm = OpenAI(
            model_name=model,
            client=None,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            best_of=best_of,
            batch_size=batch_size,
            max_retries=max_retries,
        )

    db = HeavyDB.from_env(ignore_tables=[], include_tables=None)
    print("connected to HeavyDB")
    toolkit = HeavyDBToolkit(db=db)
    agent = create_heavydb_agent(llm, toolkit, verbose=True)
    print("asking question: ", question)
    log_agent_call(agent, question, model)


if __name__ == "__main__":
    main()
