import os

import promptlayer

from heavynl.config import get_config

is_promptlayer_active = False

config = get_config()
if config.promptlayer_api_key is not None and config.promptlayer_api_key != "":
    os.environ["PROMPTLAYER_API_KEY"] = config.promptlayer_api_key
    promptlayer.api_key = config.promptlayer_api_key
    is_promptlayer_active = True


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
    if config.custom_llm is not None and config.custom_llm.type == "API":
        return config.custom_llm.api_context_window - response_tokens - 50

    model_limits = {
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 4096,
        "text-davinci": 4096,
        "gpt-4-32k": 32768,
        "gpt-4": 8192,
        "code-davinci": 8001,
    }

    for model_prefix, limit in model_limits.items():
        if model_name.startswith(model_prefix):
            return limit - response_tokens - 50

    return 4096 - response_tokens - 50
