import json
import os
import re

from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse

from heavyiq.api.handlers import (
    handle_generate_vega_spec_request,
    handle_lcel_auto_query_request,
    handle_lcel_auto_question_request,
)
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


def extract_select_columns(query: str) -> list[str]:
    """
    Extract column names from a SQL SELECT query.

    :param query: SQL query string
    :return: List of column names
    """
    # Regex pattern to extract the part between SELECT and FROM
    pattern = r"(?<=SELECT\s)(.*?)(?=\sFROM)"

    match = re.search(pattern, query, re.IGNORECASE)

    if match:
        columns, columns_part = [], match.group(1)
        for col in columns_part.split(","):
            if " AS " in col:
                column = col.split(" AS ")[1].strip()
            else:
                column = col.strip()
            columns.append(column)
        return columns
    else:
        return []


def fetch_values(query: str, session: str) -> list:
    """
    Helps to fetch VegaLite Spec Values from HeavyDB database.
    """
    from heavyiq.langchain.heavydb import HeavyDB
    from heavyiq.lcel.chains.heavydb.chart_chain import apply_limit_to_query

    db = HeavyDB.from_session(session_id=session)
    query = apply_limit_to_query(query, limit=500)
    values = db.run(query, fetch="all", to_str=False)
    select_columns = extract_select_columns(query=query)
    if not values:
        return []
    assert len(select_columns) == len(values[0])
    formatted_values = [dict(zip(select_columns, row)) for row in values]
    return formatted_values


# WebSocket for real-time communication
@chat_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_info = await chat_init()
    current_session = session_info["session"]
    SESSIONS[current_session] = websocket
    print(f"New session started: {current_session}")
    # Send session info to frontend
    await websocket.send_json({"type": "session", **session_info})
    db = None
    try:
        while True:
            message = await websocket.receive_text()
            input_dict = {"session_id": current_session, "question": message, "allowed_tables": []}
            if current_session not in HEAVYDB_INSTANCES:
                db = await HeavyDB.from_session_async(current_session)
                HEAVYDB_INSTANCES[current_session] = db
            else:
                db = HEAVYDB_INSTANCES[current_session]
            if " chart " in message:
                # invoke nl-to-vega chain
                auto_query_response = await handle_lcel_auto_query_request(input_dict, db)
                query, tables = auto_query_response.sql, auto_query_response.tables
                # send query to FE
                await websocket.send_json({"type": "sql", "message": f"{query}"})
                await websocket.send_json({"type": "processing", "message": ""})
                input_dict = {"session_id": current_session, "question": message, "query": query, "tables": tables}
                vega_response = await handle_generate_vega_spec_request(input_dict, db)
                vega_spec_json = vega_response["vega_lite_spec"]  # type: ignore
                vega_spec_dict = json.loads(vega_spec_json)
                values = fetch_values(query=query, session=current_session)
                if values:
                    vega_spec_dict["data"] = {"values": values}
                response = {"type": "chart", "message": vega_spec_dict}
            else:
                # invoke nl to answer chain
                out = await handle_lcel_auto_question_request(input_dict, db)
                await websocket.send_json({"type": "sql", "message": f"{out.sql}"})
                response = {"type": "text", "message": f"🤖 AI: {out.answer}"}  # Replace this with AI response logic
            await websocket.send_json(response)
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
