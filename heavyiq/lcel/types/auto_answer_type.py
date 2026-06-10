# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from .answer_type import AnswerChainOutputType
from .auto_sql_type import AutoSQLChainInputType


class AutoAnswerChainInputType(AutoSQLChainInputType):
    ...


class AutoAnswerChainOutputType(AnswerChainOutputType):
    tables: list[str] = Field(..., description="List of table names used.")
