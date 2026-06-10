# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import patch

# import faiss on test module is must or otherwise we should endup in segmentation fault error upon running tcs
import pytest
import sqlalchemy
from sqlalchemy import create_engine

from heavyiq.config import HeavyIQConfig

if TYPE_CHECKING:
    from heavyrag.database.base import Database


@pytest.fixture(scope="module")
def ragdb(heavyiq_config: HeavyIQConfig) -> Generator["Database", None, None]:

    with patch("heavyrag.database.base.config", heavyiq_config), patch("heavyrag.embed.CONFIG", heavyiq_config):

        from heavyiq.api import rag_initialize

        rag_initialize()
        from heavyrag.database import ragdb

        yield ragdb


@pytest.fixture(scope="function")
async def pre_clean_table(heavyiq_config: HeavyIQConfig, ragdb: "Database"):
    """
    Fixture which helps to clean/delete all records in facts table and RAG index before running each testcase.
    """
    from heavyrag.controller import rag_controller
    from heavyrag.database.models import FactsModel

    dbname = heavyiq_config.heavydb_dbname
    try:
        with ragdb.get_db() as session:
            FactsModel.delete_facts_by_database(db_session=session, heavydb_name=dbname)

        await rag_controller.delete_fact_nodes(dbname=dbname)
    except sqlalchemy.exc.OperationalError:
        pass
    yield
