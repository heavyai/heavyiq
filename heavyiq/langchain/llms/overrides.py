# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from langchain_openai import OpenAI
from langchain_community.llms.vllm import VLLMOpenAI

from heavyiq.config import get_config


class OverrideOpenAI(OpenAI):
    context_window: Optional[int] = None


class OverrideVLLMOpenAI(VLLMOpenAI):
    context_window: Optional[int] = None

    @property
    def _invocation_params(self) -> dict[str, Any]:
        config = get_config()
        invocation_params = super()._invocation_params  # type: ignore
        print(invocation_params)
        print("\n")
        if config.heavylm_api_key:
            invocation_params["extra_headers"] = {"Authorization": f"Bearer {config.heavylm_api_key}"}
        return invocation_params
