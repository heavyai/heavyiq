from __future__ import annotations
from functools import lru_cache
import os
from typing import TYPE_CHECKING

from heavyai import connect

if TYPE_CHECKING:
    from heavyai import Connection
    from heavydb._parsers import ColumnDetails

DEFAULT_TOP_K = 10

heavydb_host = os.getenv("HEAVYDB_HOST")
heavydb_port = os.getenv("HEAVYDB_PORT")
heavydb_dbname = os.getenv("HEAVYDB_DBNAME")
heavydb_username = os.getenv("HEAVYDB_USERNAME")
heavydb_password = os.getenv("HEAVYDB_PASSWORD")
heavydb_protocol = os.getenv("HEAVYDB_PROTOCOL")

conn: Connection = connect(
    user=heavydb_username, password=heavydb_password, host=heavydb_host, port=heavydb_port, dbname=heavydb_dbname
)


@lru_cache
def get_tables() -> list[str]:
    return conn.get_tables()


@lru_cache
def get_table_details(table_name: str) -> list[ColumnDetails]:
    return conn.get_table_details(table_name)


def get_table_schema(table_name: str) -> str:
    table_details: list[ColumnDetails] = get_table_details(table_name)
    res = f"CREATE TABLE {table_name} (\n"
    res += ",\n".join([f"{col.name} {col.type}" for col in table_details]) + "\n"
    res += ");\n"
    return res


def get_table_string_columns(table_name: str) -> list[str]:
    table_details = get_table_details(table_name)
    return [col.name for col in table_details if col.type == "STR" and col.encoding == "DICT"]


@lru_cache
def get_top_k_vals(table_name: str, column_name: str, top_k: int = DEFAULT_TOP_K) -> list[str | bool | int | float]:
    sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY {column_name} ORDER BY COUNT(*) DESC LIMIT {top_k}"
    results = list(conn.execute(sql))
    return [r[0] for r in results]


def get_top_k_vals_for_cols(
    table_name: str, columns: list[str], top_k: int = DEFAULT_TOP_K
) -> dict[str, list[str | bool | int | float]]:
    return {col: get_top_k_vals(table_name, col, top_k=top_k) for col in columns}
