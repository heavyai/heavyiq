# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from heavyiq.config import get_config

CONFIG = get_config()


class VectorStoreType(Enum):
    """
    Type of the vectorstore.
    """

    CHROMA = "chroma"
    FAISS = "faiss"

    @classmethod
    def from_name(cls: type["VectorStoreType"], name: str) -> "VectorStoreType":
        """
        Select an Enum value based on the provided name (case-insensitive).

        Args:
            name (str): The name of the enum value.

        Returns:
            VectorStoreType: The corresponding Enum value.
        """
        for vtype in cls:
            if vtype.value.lower() == name.lower():
                return vtype
        raise ValueError(f"VectorStoreType with name {name} does not exist")
