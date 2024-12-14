import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from async_lru import alru_cache
from cachetools import LRUCache, TTLCache, cached
from fastapi.concurrency import run_in_threadpool
from heavydb.exceptions import TDBException
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BaseChatPromptTemplate, BasePromptTemplate
from langchain.schema.cache import RETURN_VAL_TYPE, BaseCache
from langchain.schema.prompt import PromptValue
from langchain_core.tracers import langchain as langchain_tracer_module
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run

from heavyiq.config import HeavyIQConfig, change_iq_config_for_free_edition, get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.heavydb import get_db
from heavyiq.langchain.llms import LLMType, get_vllm_model_name
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.utils import TABLES_CACHE, SharedDictSingleton

is_langsmith_active = False


def init_telemetrics() -> None:
    """
    Initializes langsmith env vars only if the langchain API key exists on the config.
    """
    from heavyiq.logging_utils import get_heavyiq_logger

    global is_langsmith_active
    config = get_config()
    if config.langsmith_api_key and config.langsmith_project:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
        is_langsmith_active = True

    # chromadb telemetry opt-out
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    langchain_tracer_module.logger = get_heavyiq_logger()  # type: ignore


def enable_telemetrics_for_free_edition() -> bool:
    """
    Enables telemetrics for free edition.
    """
    if not change_iq_config_for_free_edition():
        return False

    global is_langsmith_active
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["LANGCHAIN_PROJECT"] = "heavyai-free"
    is_langsmith_active = True

    return True


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
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)
        table_info_options = [
            {"include_samples": False},
            {"include_samples": False, "include_top_k": False},
        ]

    for options in table_info_options:
        table_info = heavydb.get_table_info(table_names=table_names_to_use, **options)
        formatted_prompt = prompt.format(table_info=table_info)
        if custom_model_token_counter(formatted_prompt) <= token_limit:
            break
    else:  # executed if the loop finished normally (no break)
        raise RuntimeError("Couldn't find suitable prompt provided token limit")

    return table_info


async def aget_table_info_from_cache_or_calculate(
    session: str,
    tables: Sequence[str],
    include_samples: bool,
    include_top_k: bool,
    include_timestamp: bool,
    include_comments: bool,
) -> str:
    """
    Get tables info from cache or calculate based on the passed KW args.

    Args:
        session: valid HeavyDB session id
        table: list of tables to get info for
        include_samples: Whether to include samples on the table info. Defaults to True.
        include_top_k: Whether to include top-k on the table info. Defaults to True.
        include_timestamp: Whether to include timestmp on the table info or not. Defaults to True.
    """
    heavydb = await get_db(session)
    table_info = await heavydb.aget_table_info(
        table_names=list(tables),
        include_samples=include_samples,
        include_top_k=include_top_k,
        include_timestamp=include_timestamp,
        include_comments=include_comments,
    )
    return table_info


async def update_table_index_on_schema_change_callback(session: str, table: str):
    """
    Callback coroutine which gets executed on table schema change.
    This function helps re-genrate table document and then reindex it's metadata on chromadb vectorstore index.
    """
    from heavyiq.langchain.index.heavydb.create_index import acreate_index_if_nonexistent

    logger, shared_dict = get_heavyiq_logger(), SharedDictSingleton()  # type: ignore
    key = f"is_background_index_update_for_{table}_table_in_progress"
    has_key = await shared_dict.get(key)

    if has_key:
        logger.debug(f"An index update is already in progress for {table} table, so skipping.")
        return

    try:
        heavydb = await HeavyDB.from_session_async(session_id=session)
        await shared_dict.put(key, True)
        await asyncio.sleep(1)
        index = await acreate_index_if_nonexistent(session)
        logger.debug("Regenerating the table document and subsequently re-indexing it in ChromaDB VectorStore.")
        await index.agenerate_and_reindex_table_document(heavydb, table)
    except Exception as e:
        logger.exception(f"Failed to update index for {table} table, {e}")
    else:
        logger.debug(f"Successfully re-indexed the document for the {table} table on ChromaDB VectorStore.")
    finally:
        await shared_dict.delete(key)


TABLE_SCHEMA_ONLY_RGX: re.Pattern = re.compile(r"(?is)^create table .*?\);(?=\n|$)")
TABLE_COMMENT_RGX: re.Pattern = re.compile(r"(?is)^\s*create table (\w+)\s*(?:/\*\s*(.*?)\s*\*/)?\s*")
TABLE_COLUMN_RGX: re.Pattern = re.compile(
    r"^ *(\"[^\"]+\"|\w+)\s+(\w+(?:\[\d*\]|\([^()]*\))?(?: +ENCODING +\w+(?:\(\d*\))?)?)\s*(?:\((?:\([^()]*\)|[^()])+\))?\s*(?:/\*\s*(.*?)\s*\*/)?\s*,?\s*$"
)
COLUMN_NAME_DTYPE_RGX: re.Pattern = re.compile(
    r"^ *(\"[^\"]+\"|\w+)\s+(\w+(?:\[\d*\]|\([^()]*\))?(?: +ENCODING +\w+(?:\(\d*\))?)?)"
)
COLUMN_COMMENT_RGX: re.Pattern = re.compile(r"^.*(?:/\*\s*(.*?)\s*\*/)\s*,?\s*$")


def get_column_parts(column_detail: str) -> tuple[str, str, str | None]:
    """
    Get parts from the column detail.
    """
    name, dtype, comment = None, None, None
    match = TABLE_COLUMN_RGX.search(column_detail)
    if match:
        name, dtype, comment = match.groups()
    else:
        name, dtype = COLUMN_NAME_DTYPE_RGX.search(column_detail).groups()
        has_comment = COLUMN_COMMENT_RGX.search(column_detail)
        if has_comment:
            comment = has_comment.group(1)

    return name, dtype.strip(), comment


def retrieve_schema_details(schema: str) -> dict:
    """
    Retrieve schema details from a schema string.
    """
    # grab only the create schema stmt
    schema = TABLE_SCHEMA_ONLY_RGX.search(schema).group()
    # extract table name and optional table_comment
    table_name, table_comment = TABLE_COMMENT_RGX.search(schema).groups()
    if table_comment:
        # remove table comment from the schema after extraction
        schema = TABLE_COMMENT_RGX.sub(r"create table \1 ", schema)
    # strip column metadata like top-k and timestamp values
    lines = schema.split("\n")
    num_lines = len(lines)
    columns = []
    comment_started, intermediate_colname, intermediate_dtype, intermediate_comment = False, None, None, ""
    for i, line in enumerate(lines, start=1):
        if i == 1:
            # skip the first line
            continue
        if i == num_lines:
            line = line.split(");")[0]
        # remaining lines
        if "/*" in line and "*/" in line and not comment_started:
            # has single line comment
            name, dtype, comment = get_column_parts(line)
            columns.append((name, dtype, comment))
        elif "/*" in line and "*/" not in line and not comment_started:
            # line has hanging comment
            name, dtype = COLUMN_NAME_DTYPE_RGX.search(line).groups()
            comment_started, intermediate_colname, intermediate_dtype = True, name, dtype
            intermediate_comment += line.split("/*")[1]
        elif "/*" not in line and "*/" in line and comment_started:
            # comment end line
            intermediate_comment += "\n" + line.split("*/")[0]
            columns.append((intermediate_colname, intermediate_dtype, intermediate_comment.strip()))
            # reset to defaults
            comment_started, intermediate_colname, intermediate_dtype, intermediate_comment = False, None, None, ""
        elif comment_started:
            # comment line
            intermediate_comment += "\n" + line
        else:
            # no comment
            name, dtype = COLUMN_NAME_DTYPE_RGX.search(line).groups()
            columns.append((name, dtype, None))

    return {"name": table_name, "comment": table_comment, "columns": columns}


def is_cached_schema_equals_current_schema(cached_schema: str, current_schema: str) -> bool:
    """
    Helps to compare the table's cached schema (which may contain table and column metadata) with the curent schema (without metadata).
    """
    cached_schema_details = retrieve_schema_details(cached_schema)
    current_schema_details = retrieve_schema_details(current_schema)

    if (cached_schema_details["name"] != current_schema_details["name"]) or (
        cached_schema_details["comment"] != current_schema_details["comment"]
    ):
        return False

    cached_columns, current_columns = cached_schema_details["columns"], current_schema_details["columns"]
    if len(cached_columns) != len(current_columns):
        return False

    return cached_columns == current_columns


async def should_refresh_table_cache(heavydb: HeavyDB, table: str) -> bool:
    """
    Decides whether to refresh table cache or not.
    """
    # here key_for=tables is hardcoded because we are going to check for schema equality without top-k
    keys = TABLES_CACHE.get_table_keys(database=heavydb._dbname, table=table)
    if not keys:
        return False
    existing_data = TABLES_CACHE.get(keys[0])
    if not existing_data:
        return False

    current_data = await heavydb._aget_raw_table_schema_from_thrift(
        table, include_top_k=False, include_timestamp=False, include_comments=True, use_cache=False
    )

    return not (is_cached_schema_equals_current_schema(existing_data, current_data))


async def refresh_cache_for_tables(session: str, tables: Sequence[str]) -> None:
    """
    Check and refresh caches asscociated with the tables.
    """

    heavydb, logger, shared_dict, config = (
        await get_db(session),
        get_heavyiq_logger(),
        SharedDictSingleton(),
        get_config(),
    )

    tables_to_check = []
    for table in tables:
        last_check_tmstp = shared_dict.get_last_schema_modification_check_time(table)
        if not last_check_tmstp:
            tables_to_check.append(table)
            continue
        # do table schema check only if the difference between current_tmstp and last schema check time
        # for that table is greater than 10 mins
        if datetime.now() > (
            datetime.fromtimestamp(last_check_tmstp) + timedelta(minutes=config.table_schema_check_interval_minutes)
        ):
            tables_to_check.append(table)

    if not tables_to_check:
        return None

    logger.info(f"Checking table schema change for {tables_to_check} tables.")
    do_refresh_results = await asyncio.gather(
        *[should_refresh_table_cache(heavydb, table) for table in tables_to_check]
    )
    for table, refresh_status in zip(tables_to_check, do_refresh_results):
        shared_dict.put_last_schema_modification_check_time(table)
        if refresh_status:
            logger.info(f"Schema change detected for {table}, so resetting all the relevant caches.")
            # schema change detected
            await heavydb.delete_table_cache(table)
            TABLES_CACHE.delete_by_table_name(database=heavydb._dbname, table_name=table)
            asyncio.create_task(update_table_index_on_schema_change_callback(heavydb._conn._session, table))


async def aget_table_info_wrt_token_limit(
    llm: BaseLanguageModel,
    session: str,
    prompt: BasePromptTemplate | BaseChatPromptTemplate,
    table_names_to_use: list[str],
    caller: LLMType = LLMType.NL_TO_SQL,
) -> str:
    """
    Gets the info of all the tables with respect to the token limit async.
    """
    logger, config = get_heavyiq_logger(), get_config()
    table_names_tuple = tuple(sorted(table_names_to_use))
    # refresh-cache
    await refresh_cache_for_tables(session, table_names_to_use)

    async def is_prompt_tokens_lesser_than_token_limit(
        table_info: str, token_limit: int, token_counter: Callable
    ) -> bool:
        """
        Check whether the calculated prompt tokens lesser than the token limit allowed by the llm model.
        """
        formatted_prompt = prompt.format(table_info=table_info)
        prompt_tokens = await run_in_threadpool(token_counter, formatted_prompt)
        if prompt_tokens <= token_limit:
            return True

        return False

    # check for combined tables cache
    heavydb = await get_db(session)

    top_k_max_str_column_count = (
        config.top_k_max_str_col_count_nl_to_tables
        if caller == LLMType.NL_TO_TABLES
        else config.top_k_max_str_col_count_nl_to_sql
    )

    # decide whether to include top-k, timetstamp or not even when retriving data from cache
    disable_top_k = False
    available_text_column_count = await get_all_text_column_count(
        session, table_names_tuple
    )  # in-process cache used for upto 60 secs
    if available_text_column_count > top_k_max_str_column_count:
        disable_top_k = True

    # decides whether to include timestamp text or not
    disable_timestamp = not (config.include_timestamp_on_table_info_prompt)

    # define table/cache options
    table_info_options: list[dict[str, bool]] = [
        {},
        {"include_timestamp": False},
        {"include_timestamp": False, "include_top_k": False},
        {"include_timestamp": False, "include_top_k": False, "include_comments": False},
    ]
    if disable_timestamp and disable_top_k:
        table_info_options = [
            {"include_timestamp": False, "include_top_k": False},
            {"include_timestamp": False, "include_top_k": False, "include_comments": False},
        ]
    elif disable_top_k:
        table_info_options = [
            {"include_top_k": False},
            {"include_timestamp": False, "include_top_k": False},
            {"include_timestamp": False, "include_top_k": False, "include_comments": False},
        ]
    elif disable_timestamp:
        table_info_options = [
            {"include_timestamp": False},
            {"include_timestamp": False, "include_top_k": False},
            {"include_timestamp": False, "include_top_k": False, "include_comments": False},
        ]

    # initialize token limit and token counter
    if config.custom_llm_type is None or config.custom_llm_type == "AZURE":
        token_limit = get_token_limit(llm.model_name)  # type: ignore
        token_counter = llm.get_num_tokens
    else:
        token_limit = llm.context_window - 306  # type: ignore # (256 response + 50 buffer)
        token_counter = custom_model_token_counter

    # iterate over all the options to see which option best suits
    # cache retrieval, get table_info from cache only when the relevant cache key found for all the tables
    # else do the retrieval from database
    for options in table_info_options:
        tables_cache, table_info, found_break = [], None, False
        for table in table_names_to_use:
            key = TABLES_CACHE.create_cache_key_for_single_table(database=heavydb._dbname, table=table, **options)
            out = TABLES_CACHE.get(key)
            if not out:
                found_break = True
                break
            tables_cache.append(out)
        else:
            # cache exists for all the tables, so check if the formed table_info is lesser than the token limit
            table_info = "\n\n".join(tables_cache)
            if await is_prompt_tokens_lesser_than_token_limit(
                table_info=table_info, token_limit=token_limit, token_counter=token_counter
            ):
                logger.info(f"[Cached]: Returning table information for {table_names_to_use} tables from cache.")
                return table_info

        # don't go for other table_options if there isn't any cache found with the current option for atleast one table
        if found_break:
            break

    logger.info(f"Calculating tables_info for {table_names_to_use} tables...")

    for options in table_info_options:
        include_top_k = options.get("include_top_k", True)
        include_timestamp = options.get("include_timestamp", True)
        include_comments = options.get("include_comments", True)

        table_info = await aget_table_info_from_cache_or_calculate(
            session, table_names_tuple, False, include_top_k, include_timestamp, include_comments
        )
        # set caches for each table
        table_info_split = re.split(r"\n(?=CREATE TABLE)", table_info)
        for info in table_info_split:
            table_name = re.search(r'^CREATE TABLE "?(\w+)', info).group(1)
            single_table_cache_key = TABLES_CACHE.create_cache_key_for_single_table(
                database=heavydb._dbname,
                table=table_name,
                include_comments=include_comments,
                include_top_k=include_top_k,
                include_timestamp=include_timestamp,
            )
            out = TABLES_CACHE.get(single_table_cache_key)
            if not out:
                # populate the cache only when there isn't any key exists
                TABLES_CACHE.put(single_table_cache_key, info.strip())

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
    from tokenizers import Tokenizer

    from heavyiq.config import get_config

    config = get_config()

    model_name = get_vllm_model_name(api_base=config.custom_llm_api_base).lower()
    model_local_path: str
    if "llama-3" in model_name:
        model_local_path = "./heavyiq/langchain/tokenizer_models/llama3_model"
    elif "llama" in model_name:
        model_local_path = "./heavyiq/langchain/tokenizer_models/llama_model"
    elif "deepseek" in model_name:
        model_local_path = "./heavyiq/langchain/tokenizer_models/deepseek_model"
    elif "starcoder-2" in model_name:
        model_local_path = "./heavyiq/langchain/tokenizer_models/starcoder2_model"
    elif ("mixtral" in model_name) or ("wizard" in model_name):
        model_local_path = "./heavyiq/langchain/tokenizer_models/mixtral_model"
    elif "qwen" in model_name:
        model_local_path = "./heavyiq/langchain/tokenizer_models/qwen_model"
    else:
        raise ValueError("Unsupported model found for tokenization.")

    tok_json = f"{model_local_path}/tokenizer.json"
    tokenizer = Tokenizer.from_file(tok_json)

    tok_config_json = f"{model_local_path}/tokenizer_config.json"
    tok_config = {}
    if os.path.exists(tok_config_json):
        with open(tok_config_json, "r") as f:
            tok_config = json.load(f)

    # Add special tokens if specified in tokenizer_config
    if "bos_token" in tok_config:
        tokenizer.add_special_tokens([tok_config["bos_token"]])

    if "eos_token" in tok_config:
        tokenizer.add_special_tokens([tok_config["eos_token"]])

    return tokenizer


def custom_model_token_counter(text: str) -> int:
    """
    Token counter method for custom models which helps to calculate tokens for the given text.
    """
    return len(get_tokenizer().encode(text).tokens)


def custom_model_tokenizer_encode(text: str) -> list[int]:
    """
    Encode text str into list of input ids.
    """
    return get_tokenizer().encode(text).ids


def custom_model_tokenizer_decode(token_ids: list[int]) -> str:
    """
    Decode list of input ids back to the original text.
    """
    # skip_special_tokens=True, else <｜begin▁of▁sentence｜> gets prepended
    return get_tokenizer().decode(token_ids, skip_special_tokens=True)


@alru_cache(ttl=60)
async def get_all_text_column_count(session: str, tables: Sequence[str]) -> int:
    """
    Returns the count of all the text columns available in the tables.
    """
    tasks = []
    heavydb = await get_db(session)
    for table in tables:
        tasks.append(heavydb.aget_text_columns_count(table))

    return sum(await asyncio.gather(*tasks))


async def aget_table_info_for_nl_to_tables_prompt_wrt_token_limit(
    llm: BaseLanguageModel, session: str, prompt: BasePromptTemplate, table_names_to_use: list[str]
) -> str:
    return await aget_table_info_wrt_token_limit(llm, session, prompt, table_names_to_use, caller=LLMType.NL_TO_TABLES)


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


class NoopLangChainTracer(LangChainTracer):
    """
    Helps to disable langsmith tracing when this instance is being passed as callbacks while invoking a runnable.
    Always initialize with dummy client, so that the client validation would never happen. Ex,

    NoopLangChainTracer(client=1)
    """

    def _submit(self, function: Callable[[Run], None], run: Run) -> None:
        """Submit a function to the executor."""
        pass

    def _persist_run_single(self, run: Run) -> None:
        pass

    def _update_run_single(self, run: Run) -> None:
        pass
