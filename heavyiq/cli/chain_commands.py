from enum import Enum
from typing import Callable

import click
from fastapi.concurrency import run_in_threadpool
from langchain_community.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel

from heavyiq.cli.decorators import coro
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import (
    AskHeavyDBMetadataIndexChain,
    NLtoAnswerChain,
    SQLMetadataQuestionTransformerChain,
    get_nl_to_sql_chain_by_llm,
)
from heavyiq.langchain.index.heavydb import acreate_index_if_nonexistent, create_index_if_nonexistent
from heavyiq.langchain.index.utils import SearchType
from heavyiq.langchain.llms import LLMType, _get_custom_api_llm, _get_custom_api_vllm_llm, get_openai_llm_by_model_name
from heavyiq.langchain.logging import log_chain_call_async


class LLMCategory(Enum):
    OPENAI = "openai"
    API = "api"
    API_VLLM = "api_vllm"
    AZURE = "azure"


def validate_llm_category(ctx: click.Context, param: str, value: str) -> str:
    if value not in [item.value for item in LLMCategory]:
        raise click.BadParameter(f"Invalid choice. Choose from: {', '.join(item.value for item in LLMCategory)}")
    return value


@click.group()
def chain():
    """Run chains."""
    pass


def common_nl_to_sql_options(func: Callable) -> Callable:
    # Define common options as decorators or functions
    func = click.option(
        "--model",
        default="text-davinci-003",
        help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
        type=str,
    )(func)
    func = click.option(
        "--llm-category",
        type=click.Choice(["openai", "api", "api_vllm", "azure"]),
        default="openai",
        callback=validate_llm_category,
        help="Choose a llm category (openai/api/api_vllm/azure)",
    )(func)
    func = click.option(
        "--api-base", default="http://localhost/api/v1", help="API Base of the custom llm server", type=str
    )(func)
    func = click.option("--context-window", default=8192, help="Custom llm context window size", type=int)(func)
    func = click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)(func)
    func = click.option("--tables", default="", help="Tables to use for answer (comma-separated)", type=str)(func)
    func = click.option("--verbose", default=False, help="Verbose output", type=bool)(func)
    func = click.argument("question", type=str)(func)
    return func


async def get_llm_by_llm_category(
    model: str, temperature: float, llm_category: str, api_base: str, context_window: int
) -> BaseLLM | BaseChatModel:
    llm = None
    if llm_category in [LLMCategory.OPENAI.value, LLMCategory.AZURE.value]:
        llm = get_openai_llm_by_model_name(model=model, temperature=temperature)
    elif llm_category == LLMCategory.API.value:
        llm = _get_custom_api_llm(
            LLMType.NL_TO_SQL,
            api_base=api_base,
            context_window=context_window,
            temperature=temperature,
        )
    else:
        llm = await run_in_threadpool(
            _get_custom_api_vllm_llm,
            LLMType.NL_TO_SQL,
            api_base=api_base,
            context_window=context_window,
            temperature=temperature,
        )
    return llm


@chain.command()
@coro
@common_nl_to_sql_options
@click.pass_context  # type: ignore
async def nl_to_sql(
    ctx: click.Context,
    question: str,
    model: str,
    tables: str,
    verbose: bool,
    temperature: float,
    llm_category: str,
    api_base: str,
    context_window: int,
) -> None:
    """Call the NL to SQL Chain"""
    llm = await get_llm_by_llm_category(
        model=model,
        temperature=temperature,
        llm_category=llm_category,
        api_base=api_base,
        context_window=context_window,
    )
    heavydb = await run_in_threadpool(HeavyDB.from_env, include_tables=[t.strip() for t in tables.split(",")])
    chain = get_nl_to_sql_chain_by_llm(llm)(database=heavydb, llm=llm, verbose=verbose, tags=["cli"])  # type: ignore
    click.echo(await log_chain_call_async(chain, question, model=model))


@chain.command()
@coro
@common_nl_to_sql_options
@click.pass_context  # type: ignore
async def nl_to_answer(
    ctx: click.Context,
    question: str,
    model: str,
    tables: str,
    verbose: bool,
    temperature: float,
    llm_category: str,
    api_base: str,
    context_window: int,
) -> None:
    """Call the NL to Answer Chain"""
    llm = await get_llm_by_llm_category(
        model=model,
        temperature=temperature,
        llm_category=llm_category,
        api_base=api_base,
        context_window=context_window,
    )
    heavydb = await run_in_threadpool(HeavyDB.from_env, include_tables=[t.strip() for t in tables.split(",")])
    chain = NLtoAnswerChain.from_same_llm(llm=llm, database=heavydb, verbose=verbose, tags=["cli"])
    click.echo(await log_chain_call_async(chain, question, model))


@chain.command()
@coro
@click.option(
    "--model",
    default="text-davinci-003",
    help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
    type=str,
)
@click.option(
    "--llm-category",
    type=click.Choice(["openai", "api", "api_vllm", "azure"]),
    default="openai",
    callback=validate_llm_category,
    help="Choose a llm category (openai/api/api_vllm/azure)",
)
@click.option("--api-base", default="http://localhost/api/v1", help="API Base of the custom llm server", type=str)
@click.option("--context-window", default=8192, help="Custom llm context window size", type=int)
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.argument("question", type=str)
@click.pass_context  # type: ignore
async def question_rephraser(
    ctx: click.Context,
    question: str,
    model: str,
    temperature: float,
    llm_category: str,
    api_base: str,
    context_window: int,
) -> None:
    """Rephrase a question to be about SQL metadata"""
    llm = await get_llm_by_llm_category(
        model=model,
        temperature=temperature,
        llm_category=llm_category,
        api_base=api_base,
        context_window=context_window,
    )
    chain = SQLMetadataQuestionTransformerChain(llm=llm, tags=["cli"])
    click.echo(await log_chain_call_async(chain, question, model=model))


@chain.command()
@coro
@click.argument("question", type=str)
@click.option(
    "--search-type", default=SearchType.SIMILARITY, help="Defaults to similarity. Can also be 'mmr'.", type=SearchType
)
@click.option("--k", default=5, help="Number of Documents vector store will retrieve. Defaults to 5.", type=int)
@click.option("--fetch-k", default=20, help="Number of Documents passed to MMR algorithm. Defaults to 20.", type=int)
@click.pass_context  # type: ignore
async def ask_heavydb_index(ctx: click.Context, question: str, search_type: SearchType, k: int, fetch_k: int) -> None:
    """Ask the HeavyDB Metadata Index"""
    heavdb_metadata_index = await acreate_index_if_nonexistent()
    retriever = heavdb_metadata_index.as_retriever(search_type=search_type, k=k, fetch_k=fetch_k)
    chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
    click.echo(await log_chain_call_async(chain, question, "", chain_name="ask-heavydb-index-chain"))
