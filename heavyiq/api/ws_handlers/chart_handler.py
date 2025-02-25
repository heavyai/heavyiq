import json
import re
from datetime import datetime
from typing import AsyncGenerator

from heavyiq.api.handlers import (
    handle_generate_vega_spec_request,
    handle_lcel_auto_query_request,
    handle_lcel_auto_question_request,
    handle_vega_spec_error_correction_request,
)
from heavyiq.langchain.heavydb import HeavyDB

from .message import ChartMessage, ProcessingMessage, SQLMessage, TextMessage, WSMessage


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
        for col in re.findall(r"(?:\([^()]*\)|[^,])+", columns_part):
            if " AS " in col:
                column = col.split(" AS ")[-1].strip()
            else:
                column = col.strip()
            columns.append(column.strip('"'))
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
    # change datetime-format to string otheriwse it might endup in json decode error
    has_datetime = any(isinstance(i, datetime) for i in values[0])
    if has_datetime:
        values = [[v.isoformat() if isinstance(v, datetime) else v for v in row] for row in values]

    formatted_values = [dict(zip(select_columns, row)) for row in values]
    return formatted_values


def get_values_index(data: list) -> int | None:
    """
    Gets the Spec data item index having values as key.
    """
    for idx, item in enumerate(data):
        if "values" in item:
            return idx
        if "data" in item:
            if "values" in item["data"]:
                return idx
    return None


def attach_values_to_vega_spec(vega_spec_dict: dict, values: list) -> dict:
    """
    Attaches values data to corresponding position on vega dict.
    """
    if "data" in vega_spec_dict:
        if isinstance(vega_spec_dict["data"], list):
            # vega
            idx = get_values_index(vega_spec_dict["data"])
            assert idx is not None, "No values found on Vega Spec data"
            vega_spec_dict["data"][idx]["values"] = values
        elif isinstance(vega_spec_dict["data"], dict):
            # vega-lite
            vega_spec_dict["data"]["values"] = values
    else:
        # there might be a chances of "data" exists on the layer list
        layers = vega_spec_dict["layer"]
        idx = get_values_index(layers)
        vega_spec_dict["layer"][idx]["data"]["values"] = values

    return vega_spec_dict


def get_values_from_vega_spec(vega_spec_dict: dict) -> list:
    """
    Get values from the vega spec dict.
    """
    if "data" in vega_spec_dict:
        if isinstance(vega_spec_dict["data"], list):
            # vega
            idx = get_values_index(vega_spec_dict["data"])
            return vega_spec_dict["data"][idx]["values"]
        return vega_spec_dict["data"]["values"]

    idx = get_values_index(vega_spec_dict["layer"])
    return vega_spec_dict["layer"][idx]["data"]["values"]


async def handle_generate_chart_message(message: str, db: HeavyDB) -> AsyncGenerator[WSMessage, None]:
    """
    Handles message which contain "chart" text.
    """
    session_id = db._conn._session
    input_dict = {"session_id": session_id, "question": message, "allowed_tables": []}
    # invoke nl-to-vega chain
    auto_query_response = await handle_lcel_auto_query_request(input_dict, db)
    query, tables = auto_query_response.sql, auto_query_response.tables
    # send query to FE
    yield SQLMessage(message=query)
    yield ProcessingMessage()
    input_dict = {"session_id": session_id, "question": message, "query": query, "tables": tables}
    vega_response = await handle_generate_vega_spec_request(input_dict, db)
    vega_spec_json = vega_response["vega_lite_spec"]  # type: ignore
    # print(vega_spec_json)
    vega_spec_dict = json.loads(vega_spec_json)
    values = fetch_values(query=query, session=session_id)
    if values:
        attach_values_to_vega_spec(vega_spec_dict, values)

    yield ChartMessage(message=vega_spec_dict, new=True)


async def handle_chart_error_message(message: str, db: HeavyDB, n: int = 2) -> AsyncGenerator[WSMessage, None]:
    """
    Handles message which contain chart failed to render messages.
    "n" -> refers to the number of data rows to be included in the Vega spec as part of the prompt.
    """
    request_body, session_id = json.loads(message), db._conn._session
    vega_spec, errors = request_body["spec"], request_body["errors"]

    values_list = get_values_from_vega_spec(vega_spec)
    attach_values_to_vega_spec(vega_spec, values_list[:n])

    input_dict = {"session_id": session_id, "vega_spec": vega_spec, "errors": errors}
    response = await handle_vega_spec_error_correction_request(input_dict, db)

    corrected_spec = response["corrected_vega_spec"]  # type: ignore
    corrected_spec_dict = json.loads(corrected_spec)
    if "vega_spec" in corrected_spec_dict:
        corrected_spec_dict = corrected_spec_dict["vega_spec"]

    attach_values_to_vega_spec(corrected_spec_dict, values_list)

    yield ChartMessage(message=corrected_spec_dict, new=False)


async def handle_chart_warn_message(message: str, db: HeavyDB, n: int = 2) -> AsyncGenerator[WSMessage, None]:
    """
    Handle chart warn message.
    "n" -> n refers to the number of data rows to be included in the Vega spec as part of the prompt.
    """
    request_body, session_id = json.loads(message), db._conn._session
    vega_spec, warnings = request_body["spec"], request_body["warnings"]

    # pass only a specific data rows to error correction prompt instead of passing the whole data
    values_list = get_values_from_vega_spec(vega_spec)
    attach_values_to_vega_spec(vega_spec, values_list[:n])

    input_dict = {"session_id": session_id, "vega_spec": vega_spec, "warnings": warnings}
    response = await handle_vega_spec_error_correction_request(input_dict, db)

    corrected_spec = response["corrected_vega_spec"]  # type: ignore
    corrected_spec_dict = json.loads(corrected_spec)
    if "vega_spec" in corrected_spec_dict:
        corrected_spec_dict = corrected_spec_dict["vega_spec"]

    attach_values_to_vega_spec(corrected_spec_dict, values_list)
    yield ChartMessage(message=corrected_spec_dict, new=False)


async def handle_other_message(message: str, db: HeavyDB) -> AsyncGenerator[WSMessage, None]:
    """
    Handles all the messages which are other than charts or maps.
    """
    session_id = db._conn._session
    input_dict = {"session_id": session_id, "question": message, "allowed_tables": []}
    # invoke nl to answer chain
    out = await handle_lcel_auto_question_request(input_dict, db)
    yield SQLMessage(message=out.sql)
    yield TextMessage(message=f"🤖 AI: {out.answer}")
