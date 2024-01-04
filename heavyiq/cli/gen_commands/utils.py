import csv
import io
import math
import re
from collections import Counter
from itertools import permutations
from typing import Any, Optional

import aiocsv
import aiofiles
import numpy as np
import pandas as pd
from aiocsv.writers import AsyncWriter

from heavyiq.langchain import HeavyDB


def sql_rate_reply(
    gold_query: str, pred_query: str, db_id: str | None = None, db: HeavyDB | None = None
) -> dict[str, Any]:
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/pred
    assert db_id or db, "sql_rate_reply expects either database_name or HeavyDB instance."
    query_metadata = {
        "success": False,
        "status": "success",
        "error": None,
    }
    try:
        if gold_query == pred_query:
            query_metadata["success"] = True
            return query_metadata
        if not db:
            db = HeavyDB.from_env(db_name=db_id)
        gold_df = pd.read_sql(gold_query, db._conn)
        pred_df = pd.read_sql(pred_query, db._conn)
        num_gold_rows = len(gold_df.axes[0])  # type: ignore
        num_pred_rows = len(pred_df.axes[0])  # type: ignore
        num_gold_cols = len(gold_df.axes[1])  # type: ignore
        num_pred_cols = len(pred_df.axes[1])  # type: ignore
        if num_gold_rows != num_pred_rows:
            print("ROW COUNT MISMATCH")
            print(gold_query)
            print(pred_query)
            query_metadata["status"] = "row_count_mismatch"
            return query_metadata
        if num_gold_cols != num_pred_cols:
            print("COL COUNT MISMATCH")
            print(gold_query)
            print(pred_query)
            query_metadata["status"] = "col_count_mismatch"
            return query_metadata
        gold_query_has_order_by = gold_query.lower().find("order by") >= 0
        dfs_are_equal = None
        if gold_query_has_order_by:
            dfs_are_equal = np.array_equal(gold_df.values, pred_df.values)
        else:
            gold_df_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
            pred_df_sorted = pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True)
            dfs_are_equal = np.array_equal(gold_df_sorted.values, pred_df_sorted.values)
            # dfs_are_equal = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True).equals(pred_df.sort_values(by=list(pred_df.columns)).reset_index(drop=True))
        if dfs_are_equal:
            query_metadata["success"] = True
            return query_metadata
        else:
            for cols in permutations(pred_df.columns):
                pred_df_perm = pred_df[list(cols)]
                if gold_query_has_order_by:
                    dfs_are_equal = np.array_equal(gold_df.values, pred_df_perm.values)
                else:
                    gold_df_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
                    pred_df_perm_sorted = pred_df_perm.sort_values(by=list(pred_df_perm.columns)).reset_index(drop=True)
                    dfs_are_equal = np.array_equal(gold_df_sorted.values, pred_df_perm_sorted.values)
                if dfs_are_equal:
                    print("COLUMN ORDER DIFF")
                    query_metadata["success"] = True
                    query_metadata["status"] = "column_order_difference"
                    return query_metadata
            print("VALUES MISMATCH")
            query_metadata["status"] = "values_mismatch"
            print(gold_query)
            print(pred_query)
            return query_metadata
    except Exception as e:
        print("QUERY FAIL")
        query_metadata["status"] = "execution_error"
        query_metadata["error"] = e
        print(gold_query)
        print(pred_query)
        return query_metadata


async def awrite_gen_results_header(gen_str: str, has_id: bool):
    async with aiofiles.open(f"./gen/results/{gen_str}_queries.csv", "a", newline="") as wf:
        header = []
        if has_id:
            header.append("query_id")
        header.extend(["query_sub_id", "db_id", "question", "answer"])
        writer = AsyncWriter(wf, dialect="unix")
        await writer.writerow(header)


async def awrite_gen_results_row(
    gen_str: str,
    query_sub_id: str,
    db_id: str,
    question: str,
    answer: str,
    query_id: Optional[str] = None,
):
    async with aiofiles.open(f"./gen/results/{gen_str}_queries.csv", "a", newline="") as wf:
        writer = AsyncWriter(wf, dialect="unix")
        # Clean the data
        row_data = []
        if query_id:
            row_data.append(query_id)
        row_data.extend([query_sub_id, db_id, question, answer])
        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]
        await writer.writerow(row_data)


def extract_tables_from_query(con, query) -> list[str]:
    """
    Extract tables names from SQL query.
    """
    explain_query = "EXPLAIN CALCITE " + query
    query_plan = con.execute(explain_query)
    query_plan = list(query_plan)[0][0]
    pattern = r"LogicalTableScan\(table=\[\[(.*?)\]\]\)"
    matches = re.findall(pattern, query_plan)
    tables = []
    for match in matches:
        table = match.split(", ")[1]
        tables.append(table)
    return list(set(tables))  # remove duplicates
