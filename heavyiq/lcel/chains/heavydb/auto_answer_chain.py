# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The AutoSQL Chain module contains a runnable chain designed to generate SQL queries from natural language questions by automatically detecting the necessary tables.
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from heavyiq.lcel.types import AutoAnswerChainInputType, AutoAnswerChainOutputType

from .answer_chain import chain as answer_chain
from .table_chain import tables_dict_chain


def output(inputs: dict) -> dict:
    """
    Chain's final Output.
    """
    return {**inputs["output"], "tables": inputs["tables"], "snippet_ids": inputs["snippet_ids"]}


chain: Runnable = (
    RunnablePassthrough.assign(table_chain_response=tables_dict_chain)
    | RunnablePassthrough.assign(
        tables=lambda x: x["table_chain_response"]["tables"],
        snippet_ids=lambda x: x["table_chain_response"]["snippet_ids"],
    )
    | RunnablePassthrough.assign(output=RunnableBranch((lambda x: x["tables"], answer_chain), lambda x: {}))
    | RunnableLambda(output)
).with_types(
    input_type=AutoAnswerChainInputType, output_type=AutoAnswerChainOutputType  # type: ignore
)
