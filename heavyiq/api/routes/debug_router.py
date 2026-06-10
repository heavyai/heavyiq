# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from heavyiq.api.models.debug import DbSessionResponse
from heavyiq.langchain.heavydb import HeavyDB

debug_router = APIRouter()


@debug_router.get("/db-session", response_model=DbSessionResponse)
async def db_session(db: str | None = None) -> DbSessionResponse:
    """
    Get HeavyDB session id.
    """
    heavydb_sessionid = await HeavyDB.create_session_id_async(db_name=db)
    return DbSessionResponse(session_id=heavydb_sessionid)
