from enum import Enum
from typing import Callable

import click
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from heavyiq.cli.decorators import coro
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.llms import (
    LLMType,
    _get_custom_api_llm,
    _get_custom_api_vllm_llm,
    get_openai_llm_by_model_name,
)


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


# Note: Legacy chain CLI commands (nl_to_sql, nl_to_answer, question_rephraser, ask_heavydb_index)
# have been removed. Use the LCEL API endpoints instead:
#   - POST /api/v1/lcel/query (nl_to_sql)
#   - POST /api/v1/lcel/question (nl_to_answer)
#   - POST /api/v1/lcel/tables (table selection)
