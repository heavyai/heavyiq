# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

from pydantic import BaseModel, Field, RootModel


class QuestionsChainInputType(BaseModel):
    """Tables to Questions chain input type."""

    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list[str] = Field(
        ...,
        description="List of tables to consider.",
    )


class QuestionsChainInputTypedDict(TypedDict):
    session_id: str
    tables: list[str]


class QuestionsChainOutputType(RootModel[list[str]]):
    """Tables to Questions chain output type — list of generated questions."""
