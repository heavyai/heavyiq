# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.evaluation import EvaluationResult, RelevancyEvaluator

from heavyrag.llm import get_llm
from heavyrag.prompts import REL_EVAL_TEMPLATE, REL_REFINE_TEMPLATE

relevancy_evaluator = RelevancyEvaluator(
    llm=get_llm(), eval_template=REL_EVAL_TEMPLATE, refine_template=REL_REFINE_TEMPLATE
)


async def aevaluate_response_by_relevancy(question: str, response: RESPONSE_TYPE) -> EvaluationResult:
    """
    Evaluate response using relevancy evaluator.
    """
    relevancy_evaluator = RelevancyEvaluator(
        llm=get_llm(), eval_template=REL_EVAL_TEMPLATE, refine_template=REL_REFINE_TEMPLATE
    )
    relevancy_result = await relevancy_evaluator.aevaluate_response(query=question, response=response)
    return relevancy_result
