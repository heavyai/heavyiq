import click
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.langchain.logging import log_agent_call, log_chain_call


load_dotenv()


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


@cli.command()
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="Specify the model to use for the LLM. (Defaults to 'gpt-3.5-turbo', use 'gpt-4' for best results)",
)
@click.option("--temperature", default=0.0, type=float, help="What sampling temperature to use.")
@click.argument("question", type=str)
@click.pass_context
def agent(
    ctx: click.Context,
    model: str,
    question: str,
    temperature: float,
) -> None:
    """Call SQL Agent with a question"""
    from modules.langchain.agents.chat_agent import create_sql_agent

    chat_llm = ChatOpenAI(model_name=model, temperature=temperature, client=None)
    sql_agent = create_sql_agent(chat_llm=chat_llm)
    log_agent_call(sql_agent, question, model)


@cli.command()
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="Specify the model to use for the LLM. (Defaults to 'gpt-3.5-turbo', use 'gpt-4' for best results)",
)
@click.option("--temperature", default=0.5, type=float, help="What sampling temperature to use.")
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.pass_context
def conversational_agent(
    ctx: click.Context,
    model: str,
    verbose: bool,
    temperature: float,
) -> None:
    """Begin a conversation with an Agent with access to HeavyDB"""
    from modules.langchain.agents.convo_agent import create_conversational_agent

    chat_llm = ChatOpenAI(model_name=model, temperature=temperature, client=None)
    sql_agent = create_conversational_agent(chat_llm=chat_llm, verbose=verbose)
    # start a loop that asks for input and then calls the agent, break the loop on EXIT
    while True:
        question = input("You (type EXIT to break loop): ")
        if question == "EXIT":
            break
        # show "Processing..." until the answer returns and then show the answer in its place
        print("Processing...", end="\r")
        answer = sql_agent.run(input=question)
        print(f"Assistant: {answer}")


@cli.command()
@click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("question", type=str)
@click.pass_context
def nl_to_sql_chain(ctx: click.Context, question: str, tables: str, verbose: bool) -> None:
    """Call the NL to SQL Chain"""
    from modules.langchain.heavydb import HeavyDB
    from modules.langchain.chains.heavydb import NLtoSQLChain

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
    from modules.langchain.heavydb import HeavyDB
    from modules.langchain.chains.heavydb import NLtoAnswerChain

    heavydb = HeavyDB.from_env(include_tables=[t.strip() for t in tables.split(",")])
    llm = OpenAI(temperature=0.0, client=None)
    chain = NLtoAnswerChain(database=heavydb, llm=llm, verbose=verbose)
    click.echo(log_chain_call(chain, question, llm.model_name))


@cli.command()
@click.argument("question", type=str)
@click.pass_context
def question_rephraser_chain(ctx: click.Context, question: str) -> None:
    """Rephrase a question to be about SQL metadata"""
    from modules.langchain.chains.heavydb_index import SQLMetadataQuestionTransformerChain

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
    from modules.langchain.index import heavydb_index
    from modules.langchain.chains.heavydb_index import AskHeavyDBMetadataIndexChain

    retriever = heavydb_index.vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
    click.echo(log_chain_call(chain, question, "", chain_name="ask-heavydb-index-chain"))


if __name__ == "__main__":
    cli(obj={})
