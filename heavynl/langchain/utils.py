import os

import promptlayer
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BasePromptTemplate, BaseChatPromptTemplate

from heavynl.config import get_config
from heavynl.langchain import HeavyDB


is_promptlayer_active = False
is_langsmith_active = False


def init_promptlayer() -> None:
    """
    Initializes the promptlayer API key if it is present in the config.
    """
    global is_promptlayer_active
    config = get_config()
    if config.promptlayer_api_key is not None and config.promptlayer_api_key != "":
        os.environ["PROMPTLAYER_API_KEY"] = config.promptlayer_api_key
        promptlayer.api_key = config.promptlayer_api_key
        is_promptlayer_active = True


def init_telemetrics() -> None:
    """
    Initializes langsmith env vars only if the langchain API key exists on the config.
    """
    global is_langsmith_active
    config = get_config()
    if config.langchain_api_key and config.langchain_project:
        os.environ["LANGCHAIN_TRACING_V2"] = str(config.langchain_tracing_v2).lower()
        os.environ["LANGCHAIN_API_KEY"] = config.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
        is_langsmith_active = True


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

        tokenizer = LlamaTokenizer.from_pretrained("./heavynl/langchain/llama_model", local_files_only=True)
        token_limit = config.custom_llm_api_context_window - 306  # (256 response + 50 buffer)

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
