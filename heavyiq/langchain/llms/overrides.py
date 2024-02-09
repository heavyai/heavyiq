from typing import Any, Optional

from langchain.llms.openai import OpenAI
from langchain.llms.vllm import VLLMOpenAI

from heavyiq.config import get_config


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None


class OverrideVLLMOpenAI(VLLMOpenAI):
    context_window: Optional[int] = None

    @property
    def _invocation_params(self) -> dict[str, Any]:
        config = get_config()
        invocation_params = super()._invocation_params  # type: ignore
        if config.heavylm_api_key:
            invocation_params["headers"] = {"Authorization": f"Bearer {config.heavylm_api_key}"}
        return invocation_params
