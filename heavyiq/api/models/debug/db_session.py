# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class DbSessionResponse(BaseModel):
    session_id: str = Field(..., description="HeavyDB Session ID")
