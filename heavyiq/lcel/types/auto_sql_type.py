# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# input and output types for auto_sql_chain runnable
from pydantic import BaseModel, Field

from .sql_type import SqlChainOutputType


class AutoSQLChainInputType(BaseModel):
    """NLtoTable chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session id.")
    allowed_tables: list[str] = Field(
        default=[],
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )


class AutoSQLChainOutputType(SqlChainOutputType):
    tables: list[str] = Field(..., description="List of table names used.")
