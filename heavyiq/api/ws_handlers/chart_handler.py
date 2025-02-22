import json
import re
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
        for col in columns_part.split(","):
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
    formatted_values = [dict(zip(select_columns, row)) for row in values]
    return formatted_values


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
        vega_spec_dict["data"][0]["values"] = values
    yield ChartMessage(message=vega_spec_dict, new=True)


async def handle_chart_error_message(message: str, db: HeavyDB, n: int = 2) -> AsyncGenerator[WSMessage, None]:
    """
    Handles message which contain chart failed to render messages.
    "n" -> refers to the number of data rows to be included in the Vega spec as part of the prompt.
    """
    request_body, session_id = json.loads(message), db._conn._session
    vega_spec, errors = request_body["spec"], request_body["errors"]

    values_list = vega_spec["data"][0]["values"]
    vega_spec["data"][0]["values"] = values_list[:n]

    input_dict = {"session_id": session_id, "vega_spec": vega_spec, "errors": errors}
    response = await handle_vega_spec_error_correction_request(input_dict, db)

    corrected_spec = response["corrected_vega_spec"]
    corrected_spec_dict = json.loads(corrected_spec)
    if "vega_spec" in corrected_spec_dict:
        corrected_spec_dict = corrected_spec_dict["vega_spec"]
    corrected_spec_dict["data"][0]["values"] = values_list
    yield ChartMessage(message=corrected_spec_dict, new=False)


async def handle_chart_warn_message(message: str, db: HeavyDB, n: int = 2) -> AsyncGenerator[WSMessage, None]:
    """
    Handle chart warn message.
    "n" -> n refers to the number of data rows to be included in the Vega spec as part of the prompt.
    """
    request_body, session_id = json.loads(message), db._conn._session
    vega_spec, warnings = request_body["spec"], request_body["warnings"]

    # pass only a specific data rows to error correction prompt instead of passing the whole data
    values_list = vega_spec["data"][0]["values"]
    vega_spec["data"][0]["values"] = values_list[:n]  # pass only two data rows

    input_dict = {"session_id": session_id, "vega_spec": vega_spec, "warnings": warnings}
    response = await handle_vega_spec_error_correction_request(input_dict, db)

    corrected_spec = response["corrected_vega_spec"]
    corrected_spec_dict = json.loads(corrected_spec)
    if "vega_spec" in corrected_spec_dict:
        corrected_spec_dict = corrected_spec_dict["vega_spec"]
    corrected_spec_dict["data"][0]["values"] = values_list
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
