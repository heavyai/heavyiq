# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from heavyiq.config import get_config

from .database import Database

config = get_config()

ragdb = Database(config.rag_database_uri)
