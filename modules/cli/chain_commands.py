import click

from langchain.llms import OpenAI

from modules.langchain import HeavyDB
from modules.langchain.logging import log_chain_call
from modules.langchain.index import get_heavydb_index
from modules.langchain.chains import (
    NLtoSQLChain,
    NLtoAnswerChain,
    SQLMetadataQuestionTransformerChain,
    AskHeavyDBMetadataIndexChain,
)


@click.group()
def chain():
    """Run chains."""
    pass


@chain.command()
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_sql(ctx: click.Context, question: str, tables: str, verbose: bool) -> None:
    """Call the NL to SQL Chain"""

    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = OpenAI(temperature=0.0, client=None)
    chain = NLtoSQLChain(database=heavydb, llm=llm, verbose=verbose)
    click.echo(log_chain_call(chain, question, ""))


@chain.command()
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_answer(ctx: click.Context, question: str, tables: str, verbose: bool) -> None:
    """Call the NL to Answer Chain"""
    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = OpenAI(temperature=0.0, client=None)
    chain = NLtoAnswerChain(database=heavydb, llm=llm, verbose=verbose)
    click.echo(log_chain_call(chain, question, llm.model_name))


@chain.command()
@click.argument("question", type=str)
@click.pass_context
def question_rephraser(ctx: click.Context, question: str) -> None:
    """Rephrase a question to be about SQL metadata"""
    chain = SQLMetadataQuestionTransformerChain()
    click.echo(log_chain_call(chain, question, ""))


@chain.command()
@click.argument("question", type=str)
@click.option("--search-type", default="similarity", help="Defaults to similarity. Can also be 'mmr'.", type=str)
@click.option("--k", default=5, help="Number of Documents vector store will retrieve. Defaults to 5.", type=int)
@click.option("--fetch-k", default=20, help="Number of Documents passed to MMR algorithm. Defaults to 20.", type=int)
@click.pass_context
def ask_heavydb_index(ctx: click.Context, question: str, search_type: str, k: int, fetch_k: int) -> None:
    """Ask the HeavyDB Metadata Index"""
    retriever = get_heavydb_index().vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
    click.echo(log_chain_call(chain, question, "", chain_name="ask-heavydb-index-chain"))
