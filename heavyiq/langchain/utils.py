import os
import re
from typing import Callable

from heavydb.exceptions import TDBException
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BaseChatPromptTemplate, BasePromptTemplate
from langchain.schema.prompt import PromptValue

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB

is_langsmith_active = False


def init_telemetrics() -> None:
    """
    Initializes langsmith env vars only if the langchain API key exists on the config.
    """
    global is_langsmith_active
    config = get_config()
    if config.langsmith_api_key and config.langsmith_project:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
        is_langsmith_active = True

    # chromadb telemetry opt-out
    os.environ["ANONYMIZED_TELEMETRY"] = "false"


def get_token_limit(model_name: str, response_tokens: int = 256) -> int:
    """
    Calculate the token limit for a given model.

    Args:
        model_name (str): Name of the model.
        response_tokens (int, optional): Number of response tokens. Defaults to 256.

    Returns:
        int: Token limit for the model.

    The function checks if the model_name starts with any of the known prefixes defined in the `model_limits` dictionary,
    and subtracts the `response_tokens` and a buffer of 50 tokens from the corresponding limit.
    If the model_name does not start with any of the known prefixes, it defaults to a limit of 4096 tokens.
    If a custom API LLM is being used, `custom_llm.api_context_window` is used as the limit.
    """

    config = get_config()
    if config.custom_llm_type is not None and config.custom_llm_type == "API":
        return config.custom_llm_api_context_window - response_tokens - 50

    model_limits = {
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 4096,
        "gpt-35-turbo-16k": 16384,
        "gpt-35-turbo": 4096,
        "text-davinci": 4096,
        "text-davinci-003": 4096,
        "gpt-4-32k": 32768,
        "gpt-4": 8192,
        "code-davinci": 8001,
    }

    for model_prefix, limit in model_limits.items():
        if model_name.startswith(model_prefix):
            return limit - response_tokens - 50

    return 4096 - response_tokens - 50


def get_table_info_wrt_token_limit(
    llm: BaseLanguageModel,
    heavydb: HeavyDB,
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    table_names_to_use: list[str] | None,
) -> str:
    """
    Gets the info of all the tables with respect to the token limit.

    Args:
        table_names_to_use (list[str] | None): Get table info for the list of table names.
    """
    config = get_config()
    if config.custom_llm_type is None or config.custom_llm_type == "AZURE":
        token_limit = get_token_limit(llm.model_name)  # type: ignore
        token_counter = llm.get_num_tokens
        table_info_options: list[dict[str, bool]] = [
            {},
            {"include_top_k": False},
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]
    else:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(
            "./heavyiq/langchain/llama_model", local_files_only=True, legacy=False
        )
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)

        def token_counter(text: str) -> int:
            """Token counter for the Llama model."""
            return len(tokenizer.tokenize(text))

        table_info_options = [
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]

    for options in table_info_options:
        table_info = heavydb.get_table_info(table_names=table_names_to_use, **options)
        formatted_prompt = prompt.format(table_info=table_info)
        if token_counter(formatted_prompt) <= token_limit:
            break
    else:  # executed if the loop finished normally (no break)
        raise RuntimeError("Couldn't find suitable prompt provided token limit")

    return table_info


async def aget_table_info_wrt_token_limit(
    llm: BaseLanguageModel,
    heavydb: HeavyDB,
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    table_names_to_use: list[str] | None,
) -> str:
    """
    Gets the info of all the tables with respect to the token limit async.
    """
    config = get_config()
    if config.custom_llm_type is None or config.custom_llm_type == "AZURE":
        token_limit = get_token_limit(llm.model_name)  # type: ignore
        token_counter = llm.get_num_tokens
        table_info_options: list[dict[str, bool]] = [
            {},
            {"include_top_k": False},
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]
    else:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(
            "./heavyiq/langchain/llama_model", local_files_only=True, legacy=False
        )
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)

        def token_counter(text: str) -> int:
            """Token counter for the Llama model."""
            return len(tokenizer.tokenize(text))

        table_info_options = [
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]

    for options in table_info_options:
        table_info = await heavydb.aget_table_info(table_names=table_names_to_use, **options)
        formatted_prompt = prompt.format(table_info=table_info)
        if token_counter(formatted_prompt) <= token_limit:
            break
    else:  # executed if the loop finished normally (no break)
        raise RuntimeError("Couldn't find suitable prompt provided token limit")

    return table_info


def populate_table_info_wrt_token_limit(
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    llm: BaseLanguageModel,
    heavydb: HeavyDB,
    table_names_to_use: list[str] | None,
) -> PromptValue:
    table_info = get_table_info_wrt_token_limit(llm, heavydb, prompt, table_names_to_use)

    return prompt.format_prompt(table_info=table_info)


async def apopulate_table_info_wrt_token_limit(
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    llm: BaseLanguageModel,
    heavydb: HeavyDB,
    table_names_to_use: list[str] | None,
) -> PromptValue:
    table_info = await aget_table_info_wrt_token_limit(llm, heavydb, prompt, table_names_to_use)

    return prompt.format_prompt(table_info=table_info)


def custom_model_token_counter(text: str) -> int:
    """
    Token counter method for custom models which helps to calculate tokens for the given text.
    """
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("./heavyiq/langchain/llama_model", local_files_only=True, legacy=False)
    return len(tokenizer.tokenize(text))


async def aget_token_limit_and_token_counter_func_by_llm_with_table_options(
    llm: BaseLanguageModel,
) -> tuple[int, Callable[[str], int], list[dict[str, bool]]]:
    """
    Based on the llm type (openai or custom) and the model name, this function is
    supposed to return the token limit and the method for counting the tokens from text/prompt.
    """
    config = get_config()
    table_info_options: list[dict[str, bool]]

    if config.custom_llm_type is None or config.custom_llm_type == "AZURE":
        token_limit = get_token_limit(llm.model_name)  # type: ignore
        token_counter = llm.get_num_tokens
        table_info_options = [
            {},
            {"include_top_k": False},
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]
    else:
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)
        token_counter = custom_model_token_counter
        table_info_options = [
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]

    return token_limit, token_counter, table_info_options


# this function exactly does the job of aget_table_info_wrt_token_limit function
# but in modular form.
# TODO: remove the old function and change this func name
async def aget_table_info_for_nl_to_tables_prompt_wrt_token_limit(
    llm: BaseLanguageModel, heavydb: HeavyDB, prompt: BasePromptTemplate, table_names_to_use: list[str] | None
) -> str:
    (
        token_limit,
        token_counter,
        table_info_options,
    ) = await aget_token_limit_and_token_counter_func_by_llm_with_table_options(llm)

    for options in table_info_options:
        table_info = await heavydb.aget_table_info(table_names=table_names_to_use, **options)
        formatted_prompt = prompt.format(table_info=table_info)
        if token_counter(formatted_prompt) <= token_limit:
            break
    else:  # executed if the loop finished normally (no break)
        raise RuntimeError("Couldn't find suitable prompt provided token limit")

    return table_info


async def apopulate_table_info_into_nl_to_tables_prompt(
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    llm: BaseLanguageModel,
    heavydb: HeavyDB,
    table_names_to_use: list[str] | None,
) -> PromptValue:
    """
    Generates table info and then injects it into the given prompt.
    """
    table_info = await aget_table_info_for_nl_to_tables_prompt_wrt_token_limit(llm, heavydb, prompt, table_names_to_use)

    return prompt.format_prompt(table_info=table_info)


def extract_error_message_from_exception(exc: Exception) -> str:
    """
    Helps to extract error message from a TDBException or from a Exception instance.
    """
    if isinstance(exc, TDBException):
        return exc.error_msg

    return str(exc)
