"""
Chain for correcting the generated Vega spec.
"""

import json

from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field, Json

from heavyiq.langchain.llms import get_groq_chat_llm
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import correct_vega_lite_prompt_runnable, vega_error_chat_prompt, vega_lite_error_chat_prompt


class ChartErrorChainInputType(BaseModel):
    """
    ChartErrorChain Input type.
    """

    vega_spec: Json = Field(..., description="Vega Error Spec, ie. spec failed to render.")
    errors: list[str] = Field(default=[], description="Errors which stops the above vega to render")
    warnings: list[str] = Field(default=[], description="Warnings which stops the above vega spec to render.")


class ChartErrorChainOutputType(BaseModel):
    """
    ChartErrorChain Output type.
    """

    corrected_vega_spec: Json = Field(..., description="Corrected Vega Error Spec")


def construct_prompt_variables(inputs: dict) -> dict:
    """
    Helps to construct the prompt variables.
    """
    error_list = inputs.get("errors") or inputs.get("warnings")
    return {"error_spec": json.dumps(inputs), "error_list": error_list}


def output_formatter(values: str | AIMessage) -> dict:
    content = values.content if isinstance(values, AIMessage) else values
    if isinstance(content, str) and "```json" in content:
        content = content.strip("```json").strip("```")
    return {"corrected_vega_spec": content}


format_output_rbl = RunnableLambda(output_formatter)

# prompt_rbl = correct_vega_lite_prompt_runnable
# prompt_rbl = vega_error_chat_prompt
prompt_rbl = vega_lite_error_chat_prompt
# llm_rbl = llm_runnable.with_config(configurable={"llm": "default_llm", "llm_max_tokens": 1024}).bind(
#     extra_body={"guided_regex": "{.*}"}
# )
llm_rbl = get_groq_chat_llm()

chain = (
    (construct_prompt_variables | prompt_rbl | llm_rbl | format_output_rbl)
    .with_types(input_type=ChartErrorChainInputType, output_type=ChartErrorChainOutputType)  # type: ignore
    .with_config(
        config={"tags": ["FixVegaLiteSpecChainRunnable"], "run_name": "Fix Vega Lite Spec Chain"}  # type: ignore
    )
)
