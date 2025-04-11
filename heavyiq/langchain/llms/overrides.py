from typing import Any, Optional

from langchain_community.llms.vllm import VLLMOpenAI
from langchain_openai import OpenAI

from heavyiq.config import get_config
from heavyiq.logging_utils import get_heavyiq_logger


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None


class OverrideVLLMOpenAI(VLLMOpenAI):
    context_window: Optional[int] = None

    @property
    def _invocation_params(self) -> dict[str, Any]:
        config, logger = get_config(), get_heavyiq_logger()
        invocation_params = super()._invocation_params  # type: ignore
        logger.info(f"Calling the vLLM model with the following parameters:\n{invocation_params}")
        if config.heavylm_api_key:
            invocation_params["extra_headers"] = {"Authorization": f"Bearer {config.heavylm_api_key}"}
        return invocation_params
