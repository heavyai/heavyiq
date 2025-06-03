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
from heavyiq.api.ws_handlers.message import ChartMessage, WSMessage
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


chart_data_1 = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 800,
    "height": 600,
    "projection": {"type": "mercator"},
    "layer": [
        {
            "data": {
                "url": "https://vega.github.io/vega-lite/data/world-110m.json",
                "format": {"type": "topojson", "feature": "countries"},
            },
            "mark": {"type": "geoshape", "fill": "lightgray", "stroke": "white"},
        },
        {
            "data": {
                "values": [
                    {"longitude": -74.006, "latitude": 40.7128, "city": "New York"},
                    {"longitude": -118.2437, "latitude": 34.0522, "city": "Los Angeles"},
                    {"longitude": -0.1278, "latitude": 51.5074, "city": "London"},
                ]
            },
            "mark": "circle",
            "encoding": {
                "longitude": {"field": "longitude", "type": "quantitative"},
                "latitude": {"field": "latitude", "type": "quantitative"},
                "size": {"value": 100},
                "color": {"value": "red"},
                "tooltip": {"field": "city", "type": "nominal"},
            },
        },
    ],
}

chart_data_2 = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 800,
    "height": 600,
    "projection": {"type": "mercator"},
    "layer": [
        {
            "data": {
                "url": "https://vega.github.io/vega-lite/data/us-10m.json",
                "format": {"type": "topojson", "feature": "states"},
            },
            "transform": [{"filter": "datum.id == '48'"}],
            "mark": {"type": "geoshape", "fill": "lightgray", "stroke": "white"},
        },
        {
            "data": {
                "values": [
                    {"longitude": -98.5, "latitude": 28.0, "well_name": "Well A"},
                    {"longitude": -98.2, "latitude": 27.9, "well_name": "Well B"},
                    {"longitude": -97.8, "latitude": 28.1, "well_name": "Well C"},
                ]
            },
            "mark": "circle",
            "encoding": {
                "longitude": {"field": "longitude", "type": "quantitative"},
                "latitude": {"field": "latitude", "type": "quantitative"},
                "size": {"value": 100},
                "color": {"value": "red"},
                "tooltip": {"field": "well_name", "type": "nominal"},
            },
        },
    ],
}


chart_data_3 = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 800,
    "height": 600,
    "projection": {"type": "mercator"},
    "data": {
        "url": "https://vega.github.io/vega-lite/data/world-110m.json",
        "format": {"type": "topojson", "feature": "countries"},
    },
    "transform": [
        {
            "lookup": "id",
            "from": {
                "data": {
                    "values": [
                        {"id": "840", "population": 331002651},
                        {"id": "156", "population": 1439323776},
                        {"id": "356", "population": 1380004385},
                        {"id": "76", "population": 212559417},
                        {"id": "643", "population": 145912025},
                        {"id": "124", "population": 37742154},
                        {"id": "250", "population": 65273511},
                        {"id": "36", "population": 25499884},
                        {"id": "710", "population": 59308690},
                    ]
                },
                "key": "id",
                "fields": ["population"],
            },
        }
    ],
    "mark": "geoshape",
    "encoding": {
        "color": {"field": "population", "type": "quantitative", "scale": {"scheme": "blues"}},
        "tooltip": [
            {"field": "id", "type": "nominal", "title": "Country ID"},
            {"field": "population", "type": "quantitative", "title": "Population"},
        ],
    },
}

chart_3 = {
    "$schema": "https://vega.github.io/schema/vega/v5.json",
    "description": "A choropleth map depicting U.S. unemployment rates by county in 2009.",
    "width": 960,
    "height": 500,
    "autosize": "none",
    "data": [
        {
            "name": "unemp",
            "url": "https://raw.githubusercontent.com/vega/vega/main/docs/data/unemployment.tsv",
            "format": {"type": "tsv", "parse": "auto"},
        },
        {
            "name": "counties",
            "url": "https://raw.githubusercontent.com/vega/vega/main/docs/data/us-10m.json",
            "format": {"type": "topojson", "feature": "counties"},
            "transform": [
                {"type": "lookup", "from": "unemp", "key": "id", "fields": ["id"], "values": ["rate"]},
                {"type": "filter", "expr": "datum.rate != null"},
            ],
        },
    ],
    "projections": [{"name": "projection", "type": "albersUsa"}],
    "scales": [{"name": "color", "type": "quantize", "domain": [0, 0.15], "range": {"scheme": "blues", "count": 7}}],
    "legends": [{"fill": "color", "orient": "bottom-right", "title": "Unemployment", "format": "0.1%"}],
    "marks": [
        {
            "type": "shape",
            "from": {"data": "counties"},
            "encode": {
                "enter": {"tooltip": {"signal": "format(datum.rate, '0.1%')"}},
                "update": {"fill": {"scale": "color", "field": "rate"}},
                "hover": {"fill": {"value": "red"}},
            },
            "transform": [{"type": "geoshape", "projection": "projection"}],
        }
    ],
}

bar = {
    "$schema": "https://vega.github.io/schema/vega/v5.json",
    "width": 800,
    "height": 800,
    "padding": 50,
    "data": [
        {
            "name": "table",
            "values": [
                {"weight": 4295, "horsepower": 130},
                {"weight": 3520, "horsepower": 110},
                {"weight": 3425, "horsepower": 105},
                {"weight": 3630, "horsepower": 100},
                {"weight": 3525, "horsepower": 98},
            ],
        }
    ],
    "scales": [
        {
            "name": "xscale",
            "type": "linear",
            "domain": {"data": "table", "field": "horsepower"},
            "range": [0, {"signal": "width"}],
        },
        {
            "name": "yscale",
            "type": "linear",
            "domain": {"data": "table", "field": "weight"},
            "range": [{"signal": "height"}, 0],
        },
    ],
    "marks": [
        {
            "type": "symbol",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "horsepower"},
                    "y": {"scale": "yscale", "field": "weight"},
                    "fill": {"value": "steelblue"},
                    "size": {"value": 50},
                }
            },
        }
    ],
    "axes": [
        {"scale": "xscale", "orient": "bottom", "title": "Horsepower"},
        {"scale": "yscale", "orient": "left", "title": "Weight"},
    ],
}


async def process_ws_message(message: str, db: HeavyDB) -> AsyncGenerator[WSMessage, None]:
    """Routes WebSocket messages to the appropriate handler."""
    # if "Map " or " map " or "map " in message:
    #     yield ChartMessage(message=bar, new=True)
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
        await websocket.close()
        print(f"Session {current_session} ended.")
