# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypedDict


class StepDict(TypedDict):
    event: Literal["step", "output", "error"]
    data: str
