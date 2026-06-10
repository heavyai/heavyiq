# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .utils import get_latest_heavyai_docs
from ..utils import HeavyIQIndexWrapper


class HeavyAIDocsIndex(HeavyIQIndexWrapper):
    """Index for the Heavy.AI Immerse documentation."""

    def update_docs(self):
        """Removes all documentation from the vector store and redownloads/reindexes the newest immerse docs on the HEAVY.AI site."""
        documents = get_latest_heavyai_docs()
        self.vectorstore._collection.delete()
        self.vectorstore.add_documents(documents)
