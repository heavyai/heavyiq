import json
import os
import re
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse

from heavyiq.api.handlers import (
    handle_generate_vega_spec_request,
    handle_lcel_auto_query_request,
    handle_lcel_auto_question_request,
)
from heavyiq.api.ws_handlers import (
    handle_chart_error_message,
    handle_chart_warn_message,
    handle_generate_chart_message,
    handle_other_message,
)
from heavyiq.api.ws_handlers.message import WSMessage
from heavyiq.langchain import HeavyDB
from heavyiq.logging_utils import heavyiq_logger as logger

chat_router = APIRouter()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
TEMPLATE_PATH = os.path.join(BASE_DIR, "../../templates/chat.html")  # Adjust path

# Dictionary to store active sessions
SESSIONS = {}
HEAVYDB_INSTANCES = {}


# Serve the Chat UI
@chat_router.get("")
async def chat_ui() -> FileResponse:
    return FileResponse(TEMPLATE_PATH)


async def chat_init() -> dict[str, str]:
    db: HeavyDB = await HeavyDB.create_with_persistant_connection_async()
    return {"database": db._dbname, "session": db._conn._session}


async def process_ws_message(message: str, db: HeavyDB) -> AsyncGenerator[WSMessage, None]:
    """Routes WebSocket messages to the appropriate handler."""
    if " chart " in message:
        logger.info("WS: Generate Chart message received!")
        async for response in handle_generate_chart_message(message, db):
            yield response
    elif "chart_error" in message:
        logger.info("WS: Fix Chart w.r.t error message received!")
        async for response in handle_chart_error_message(message, db):
            yield response
    elif "chart_warning" in message:
        logger.info("WS: Fix Chart w.r.t warn message received!")
        async for response in handle_chart_warn_message(message, db):
            yield response
    else:
        logger.info("WS: NL to Answer message received!")
        async for response in handle_other_message(message, db):
            yield response


# WebSocket for real-time communication
@chat_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_info = await chat_init()
    current_session, current_database = session_info["session"], session_info["database"]
    SESSIONS[current_session] = websocket
    logger.info(f"New Chat session started: Session ID: {current_session}, Database: {current_database}")
    # Send session info to frontend
    await websocket.send_json({"type": "session", **session_info})
    db = None
    try:
        if current_session not in HEAVYDB_INSTANCES:
            db = await HeavyDB.from_session_async(current_session)
            HEAVYDB_INSTANCES[current_session] = db
        else:
            db = HEAVYDB_INSTANCES[current_session]
        while True:
            message = await websocket.receive_text()
            async for response in process_ws_message(message, db):
                await websocket.send_json(response.dict())
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Session {current_session} closed: {e}")
    finally:
        # Remove session on disconnect
        del SESSIONS[current_session]
        if db:
            db._conn.close()
        print(f"Session {current_session} ended.")
