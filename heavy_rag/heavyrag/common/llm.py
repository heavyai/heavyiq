import json
from typing import Any, Dict

import requests
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.llms.vllm import VllmServer
from llama_index.llms.vllm.utils import post_http_request


def get_completion_response(response: requests.Response) -> str:
    choices = json.loads(response.content)["choices"]
    if choices:
        return choices[0]["text"]
    return ""


class OverrideVllmServer(VllmServer):
    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "n": self.n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "use_beam_search": self.use_beam_search,
            "best_of": self.best_of,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {**base_kwargs}

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        kwargs = kwargs if kwargs else {}
        params = {**self._model_kwargs, **kwargs}

        # build sampling parameters
        sampling_params = dict(**params)
        sampling_params["prompt"] = prompt
        response = post_http_request(self.api_url, sampling_params, stream=False)
        output = get_completion_response(response)

        return CompletionResponse(text=output)
