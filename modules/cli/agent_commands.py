import click

from langchain.chat_models import ChatOpenAI

from modules.langchain.agents.chat_agent import create_sql_agent
from modules.langchain.agents.convo_agent import create_conversational_agent
from modules.langchain.logging import log_agent_call


@click.group()
def agent():
    """Run agents."""
    pass


@agent.command()
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="Specify the model to use for the LLM. (Defaults to 'gpt-3.5-turbo', use 'gpt-4' for best results)",
)
@click.option("--temperature", default=0.0, type=float, help="What sampling temperature to use.")
@click.argument("question", type=str)
@click.pass_context
def one_time(
    ctx: click.Context,
    model: str,
    question: str,
    temperature: float,
) -> None:
    """Call SQL Agent with a question"""

    chat_llm = ChatOpenAI(model_name=model, temperature=temperature, client=None)
    sql_agent = create_sql_agent(chat_llm=chat_llm)
    log_agent_call(sql_agent, question, model)


@agent.command()
@click.option(
    "--model",
    default="gpt-3.5-turbo",
    help="Specify the model to use for the LLM. (Defaults to 'gpt-3.5-turbo', use 'gpt-4' for best results)",
)
@click.option("--temperature", default=0.5, type=float, help="What sampling temperature to use.")
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.pass_context
def conversational(
    ctx: click.Context,
    model: str,
    verbose: bool,
    temperature: float,
) -> None:
    """Begin a conversation with an Agent with access to HeavyDB"""

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
