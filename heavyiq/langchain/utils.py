import asyncio
import os
from typing import Any, Callable, Sequence

from async_lru import alru_cache
from cachetools import LRUCache, TTLCache, cached
from fastapi.concurrency import run_in_threadpool
from heavydb.exceptions import TDBException
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BaseChatPromptTemplate, BasePromptTemplate
from langchain.schema.cache import RETURN_VAL_TYPE, BaseCache
from langchain.schema.prompt import PromptValue

from heavyiq.config import HeavyIQConfig, get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import LLMType, get_vllm_model_name
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.utils import SharedDictSingleton

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


@cached(cache=TTLCache(maxsize=30, ttl=60 * 10))
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
        tokenizer = get_tokenizer()
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


@alru_cache(maxsize=127, ttl=60 * 10)  # typed=True and passing kwargs seems buggy in async lru
async def aget_table_info_from_cache_or_calculate(
    session: str, tables: Sequence[str], include_samples: bool, include_top_k: bool
) -> str:
    """
    Get tables info from cache or calculate based on the passed KW args.

    Args:
        session: valid HeavyDB session id
        table: list of tables to get info for
        include_samples: Whether to include samples on the table info. Defaults to True.
        include_top_k: Whether to include top-k on the table info. Defaults to True.
    """
    heavydb = await get_db(session)
    table_info = await heavydb.aget_table_info(
        table_names=list(tables), include_samples=include_samples, include_top_k=include_top_k
    )
    return table_info


async def update_table_index_on_schema_change_callback(heavydb: HeavyDB, table: str):
    """
    Callback coroutine which gets executed on table schema change.
    This function helps re-genrate table document and then reindex it's metadata on chromadb vectorstore index.
    """
    from heavyiq.langchain.index.heavydb import aget_heavydb_index

    logger, shared_dict = get_heavyiq_logger(), SharedDictSingleton()  # type: ignore
    key = f"is_background_index_update_for_{table}_table_in_progress"
    has_key = await shared_dict.get(key)

    if has_key:
        logger.debug(f"An index update is already in progress for {table} table, so skipping.")
        return

    try:
        await shared_dict.put(key, True)
        await asyncio.sleep(1)
        index = await aget_heavydb_index()
        logger.debug("Regenerating the table document and subsequently re-indexing it in ChromaDB VectorStore.")
        await index.agenerate_and_reindex_table_document(heavydb, table)
    except Exception as e:
        logger.exception(f"Failed to update index for {table} table, {e}")
    else:
        logger.debug(f"Successfully re-indexed the document for the {table} table on ChormaDB VectorStore.")
    finally:
        await shared_dict.delete(key)


@alru_cache(maxsize=127, ttl=60 * 10)
async def refresh_cache_for_tables(session: str, tables: Sequence[str]):
    """
    Check and refresh caches asscociated with the tables.
    """
    heavydb = await get_db(session)
    do_refresh_results = await asyncio.gather(*[heavydb.should_refresh_table_cache(table) for table in tables])
    for table, refresh_status in zip(tables, do_refresh_results):
        if refresh_status:
            # schema change detected
            await asyncio.gather(
                heavydb.delete_table_cache(table),
                heavydb.trigger_table_schema_change_callback(
                    table, callback=update_table_index_on_schema_change_callback
                ),
            )

    # if any of the table schema gets changed, then clear it's table_info cache
    if True in do_refresh_results:
        options_list = [(True, True), (True, False), (False, True), (False, False)]
        for x, y in options_list:
            aget_table_info_from_cache_or_calculate.cache_invalidate(heavydb._conn._session, tables, x, y)


async def aget_table_info_wrt_token_limit(
    llm: BaseLanguageModel,
    session: str,
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    table_names_to_use: list[str] | None,
    caller: LLMType = LLMType.NL_TO_SQL,
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
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)
        token_counter = custom_model_token_counter
        table_info_options = [
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]

    top_k_max_str_column_count = (
        config.top_k_max_str_col_count_nl_to_tables
        if caller == LLMType.NL_TO_TABLES
        else config.top_k_max_str_col_count_nl_to_sql
    )

    # refresh-cache
    table_names_tuple = tuple(table_names_to_use)
    await refresh_cache_for_tables(session, table_names_tuple)

    # decides whether to include top-k or not
    disable_top_k = False
    available_text_column_count = await get_all_text_column_count(session, table_names_tuple)
    if available_text_column_count > top_k_max_str_column_count:
        disable_top_k = True

    for options in table_info_options:
        include_samples = options.get("include_samples", True)
        include_top_k = options.get("include_top_k", True)
        if disable_top_k:
            include_top_k = False

        table_info = await aget_table_info_from_cache_or_calculate(
            session, table_names_tuple, include_samples, include_top_k
        )
        formatted_prompt = prompt.format(table_info=table_info)
        prompt_tokens = await run_in_threadpool(token_counter, formatted_prompt)
        if prompt_tokens <= token_limit:
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
    table_info = await aget_table_info_wrt_token_limit(llm, heavydb._conn._session, prompt, table_names_to_use)

    return prompt.format_prompt(table_info=table_info)


@cached(cache=LRUCache(maxsize=3))  # Cache the tokenizer
def get_tokenizer() -> Any:
    """
    Get tokenizer from pretrained local files.
    """
    from transformers import AutoTokenizer, LlamaTokenizer

    from heavyiq.config import get_config

    config = get_config()

    model_name = get_vllm_model_name(api_base=config.custom_llm_api_base)
    model_local_path: str
    if "llama" in model_name:
        tokenizer_cls = LlamaTokenizer
        model_local_path = "./heavyiq/langchain/tokenizer_models/llama_model"
    elif "deepseek" in model_name:
        tokenizer_cls = AutoTokenizer
        model_local_path = "./heavyiq/langchain/tokenizer_models/deepseek_model"
    else:
        raise ValueError("Unsupported model found for tokenization. Aavailable models are 'llama' and 'deepseek'.")

    return tokenizer_cls.from_pretrained(model_local_path, local_files_only=True, legacy=False)


def custom_model_token_counter(text: str) -> int:
    """
    Token counter method for custom models which helps to calculate tokens for the given text.
    """
    return len(get_tokenizer().tokenize(text))


def custom_model_tokenizer_encode(text: str) -> list[int]:
    """
    Encode text str into list of input ids.
    """
    return get_tokenizer().encode(text)


def custom_model_tokenizer_decode(token_ids: list[int]) -> str:
    """
    Decode list of input ids back to the original text.
    """
    # skip_special_tokens=True, else <｜begin▁of▁sentence｜> gets prepended
    return get_tokenizer().decode(token_ids, skip_special_tokens=True)


async def aget_token_limit_and_token_counter_func_by_llm_with_table_options(
    llm: BaseLanguageModel,
) -> tuple[int, Callable[[str], int], int, list[dict[str, bool]]]:
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

    top_k_max_str_column_count = config.top_k_max_str_col_count_nl_to_tables

    return token_limit, token_counter, top_k_max_str_column_count, table_info_options


@alru_cache(maxsize=127, ttl=60 * 10)
async def get_all_text_column_count(session: str, tables: Sequence[str]) -> int:
    """
    Returns the count of all the text columns available in the tables.
    """
    tasks = []
    heavydb = await get_db(session)
    for table in tables:
        tasks.append(heavydb.aget_text_columns_count(table))

    return sum(await asyncio.gather(*tasks))


# this function exactly does the job of aget_table_info_wrt_token_limit function
# but in modular form.
# TODO: remove the old function and change this func name
async def aget_table_info_for_nl_to_tables_prompt_wrt_token_limit(
    llm: BaseLanguageModel, session: str, prompt: BasePromptTemplate, table_names_to_use: list[str]
) -> str:
    (
        token_limit,
        token_counter,
        top_k_max_str_column_count,
        table_info_options,
    ) = await aget_token_limit_and_token_counter_func_by_llm_with_table_options(llm)

    # refresh-cache
    table_names_tuple = tuple(table_names_to_use)
    await refresh_cache_for_tables(session, table_names_tuple)

    # decides whether to include top-k or not
    disable_top_k = False
    available_text_column_count = await get_all_text_column_count(session, table_names_tuple)
    if available_text_column_count > top_k_max_str_column_count:
        disable_top_k = True

    for options in table_info_options:
        include_samples = options.get("include_samples", True)
        include_top_k = options.get("include_top_k", True)
        if disable_top_k:
            include_top_k = False
        table_info = await aget_table_info_from_cache_or_calculate(
            session, table_names_tuple, include_samples, include_top_k
        )
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
    table_info = await aget_table_info_for_nl_to_tables_prompt_wrt_token_limit(
        llm, heavydb._conn._session, prompt, table_names_to_use
    )

    return prompt.format_prompt(table_info=table_info)


def extract_error_message_from_exception(exc: Exception) -> str:
    """
    Helps to extract error message from a TDBException or from a Exception instance.
    """
    if isinstance(exc, TDBException):
        return exc.error_msg

    return str(exc)


class InMemoryLLMCache(BaseCache):
    """
    Custom in-memory cache class used to store LLM results.
    """

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}
        self._config: HeavyIQConfig = get_config()

    def _is_instruct_prompt(self, prompt: str) -> bool:
        """
        Checks whether the prompt is built up for INSTRUCT model or not.
        """
        if prompt.startswith(self._config.custom_llm_api_instruct_prompt_start_token):
            return True
        return False

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on prompt and llm_string."""
        if self._is_instruct_prompt(prompt):
            return self._cache.get((prompt, llm_string), None)
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        if self._is_instruct_prompt(prompt):
            self._cache[(prompt, llm_string)] = return_val
        return None

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}
