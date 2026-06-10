# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum


class LangChainType(enum.Enum):
    Agent = "Agent"
    Chain = "Chain"
    RunnableChain = "RunnableChain"
