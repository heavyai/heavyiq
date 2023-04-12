import click
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.langchain.logging import log_agent_call, log_chain_call
from modules.langchain.heavydb import HeavyDB
from modules.langchain.chains.heavydb import NLtoSQLChain, NLtoAnswerChain
from modules.langchain.chains.heavydb_index import SQLMetadataQuestionTransformerChain, AskHeavyDBMetadataIndexChain
from modules.langchain.agents.agent_toolkits.heavydb.base import create_heavydb_agent
from modules.langchain.agents.agent_toolkits.heavydb.toolkit import HeavyDBToolkit
from modules.langchain.index import heavydb_index

load_dotenv()


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


@cli.command()
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
@click.argument("question", type=str)
@click.pass_context
def agent(
    ctx: click.Context,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    n: int,
    best_of: int,
    batch_size: int,
    max_retries: int,
) -> None:
    """Call SQL Agent with a question"""
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
    click.echo("connected to HeavyDB")
    toolkit = HeavyDBToolkit(db=db)
    agent = create_heavydb_agent(llm, toolkit, verbose=True)
    click.echo(f"asking question: {question}")
    log_agent_call(agent, question, model)


@cli.command()
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_sql_chain(ctx: click.Context, question: str, tables: str, verbose: bool) -> None:
    """Call the NL to SQL Chain"""
    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = OpenAI(temperature=0.0, client=None)
    chain = NLtoSQLChain(database=heavydb, llm=llm, verbose=verbose)
    click.echo(log_chain_call(chain, question, ""))


@cli.command()
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_answer_chain(ctx: click.Context, question: str, tables: str, verbose: bool) -> None:
    """Call the NL to Answer Chain"""
    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = OpenAI(temperature=0.0, client=None)
    chain = NLtoAnswerChain(database=heavydb, llm=llm, verbose=verbose)
    click.echo(log_chain_call(chain, question, ""))


@cli.command()
@click.argument("question", type=str)
@click.pass_context
def question_rephraser_chain(ctx: click.Context, question: str) -> None:
    """Rephrase a question to be about SQL metadata"""
    chain = SQLMetadataQuestionTransformerChain()
    click.echo(log_chain_call(chain, question, ""))


@cli.command()
@click.argument("question", type=str)
@click.option("--search-type", default="similarity", help="Defaults to similarity. Can also be 'mmr'.", type=str)
@click.option("--k", default=5, help="Number of Documents vector store will retrieve. Defaults to 5.", type=int)
@click.option("--fetch-k", default=20, help="Number of Documents passed to MMR algorithm. Defaults to 20.", type=int)
@click.pass_context
def ask_heavydb_index_chain(ctx: click.Context, question: str, search_type: str, k: int, fetch_k: int) -> None:
    """Ask the HeavyDB Metadata Index"""
    retriever = heavydb_index.vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
    click.echo(log_chain_call(chain, question, "", chain_name="ask-heavydb-index-chain"))


if __name__ == "__main__":
    cli(obj={})
