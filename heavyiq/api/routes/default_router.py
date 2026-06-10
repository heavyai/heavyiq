# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

defaultrouter = APIRouter()

PUBLIC_PATH = "public"


@defaultrouter.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@defaultrouter.get("/version.txt")
async def serve_version() -> FileResponse:
    file_path = os.path.join(PUBLIC_PATH, "version.txt")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Not Found")

    return FileResponse(file_path)
