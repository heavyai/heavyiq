import click

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.logging import log_chain_call
from heavyiq.langchain.index.heavydb import SearchType, get_heavydb_index
from heavyiq.langchain.chains import (
    NLtoAnswerChain,
    SQLMetadataQuestionTransformerChain,
    AskHeavyDBMetadataIndexChain,
    get_nl_to_sql_chain_by_llm,
)
from heavyiq.langchain.llms import get_llm_by_model_name


@click.group()
def chain():
    """Run chains."""
    pass


@chain.command()
@click.option(
    "--model",
    default="text-davinci-003",
    help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
)
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_sql(ctx: click.Context, question: str, model: str, tables: str, verbose: bool, temperature: float) -> None:
    """Call the NL to SQL Chain"""

    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = get_llm_by_model_name(model, temperature=temperature, client=None)
    chain = get_nl_to_sql_chain_by_llm(llm)(database=heavydb, llm=llm, verbose=verbose, tags=["cli"])  # type: ignore
    click.echo(log_chain_call(chain, question, ""))


@chain.command()
@click.option(
    "--model",
    default="text-davinci-003",
    help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
)
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_answer(ctx: click.Context, question: str, model: str, tables: str, verbose: bool, temperature: float) -> None:
    """Call the NL to Answer Chain"""
    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = get_llm_by_model_name(model, temperature=temperature, client=None)
    chain = NLtoAnswerChain.from_same_llm(llm=llm, database=heavydb, verbose=verbose, tags=["cli"])
    click.echo(log_chain_call(chain, question, model))


@chain.command()
@click.option(
    "--model",
    default="text-davinci-003",
    help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
)
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.argument("question", type=str)
@click.pass_context
def question_rephraser(ctx: click.Context, question: str, model: str, temperature: float) -> None:
    """Rephrase a question to be about SQL metadata"""
    llm = get_llm_by_model_name(model, temperature=temperature, client=None)
    chain = SQLMetadataQuestionTransformerChain(llm=llm, tags=["cli"])
    click.echo(log_chain_call(chain, question, ""))


@chain.command()
@click.argument("question", type=str)
@click.option(
    "--search-type", default=SearchType.SIMILARITY, help="Defaults to similarity. Can also be 'mmr'.", type=SearchType
)
@click.option("--k", default=5, help="Number of Documents vector store will retrieve. Defaults to 5.", type=int)
@click.option("--fetch-k", default=20, help="Number of Documents passed to MMR algorithm. Defaults to 20.", type=int)
@click.pass_context
def ask_heavydb_index(ctx: click.Context, question: str, search_type: SearchType, k: int, fetch_k: int) -> None:
    """Ask the HeavyDB Metadata Index"""
    retriever = get_heavydb_index().as_retriever(search_type=search_type, k=k, fetch_k=fetch_k)
    chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
    click.echo(log_chain_call(chain, question, "", chain_name="ask-heavydb-index-chain"))
