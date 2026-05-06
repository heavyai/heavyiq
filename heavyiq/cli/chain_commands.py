from enum import Enum

import click
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from heavyiq.cli.decorators import coro
from heavyiq.langchain.chains import AskHeavyDBMetadataIndexChain, SQLMetadataQuestionTransformerChain
from heavyiq.langchain.index.heavydb import acreate_index_if_nonexistent
from heavyiq.langchain.index.utils import SearchType
from heavyiq.langchain.llms import (
    LLMType,
    _get_custom_api_llm,
    _get_custom_api_vllm_llm,
    get_openai_llm_by_model_name,
)
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
