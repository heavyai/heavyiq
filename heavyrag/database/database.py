# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class Database:
    def __init__(self, database_uri: str):
        self.database_uri = database_uri
        self.engine = create_engine(self.database_uri)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create_tables(self):
        # make sure to create db parent if not exists
        parent_dir = re.search(r"sqlite:///(.*)/[^/]+\.sqlite", self.database_uri).group(1)
        os.makedirs(parent_dir, exist_ok=True)
        Base.metadata.create_all(bind=self.engine)
        logger.info("Created tables in the database")
