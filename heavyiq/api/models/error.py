# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """

    error: str = Field(..., description="An error message")

    class Config:
        json_schema_extra = {
            "example": {"error": "Internal Server Error"},
        }
