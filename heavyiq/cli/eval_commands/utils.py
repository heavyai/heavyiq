from typing import Any, Optional
from itertools import permutations
from collections import Counter
import io
import csv

import pandas as pd
import numpy as np

from heavyiq.langchain import HeavyDB


def sql_rate_reply(db_id: str, gold_query: str, pred_query: str) -> dict[str, Any]:
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/pred
    query_metadata = {
        "success": False,
        "status": "success",
        "error": None,
    }
    try:
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


def write_eval_results_header(eval_str: str, has_id: bool):
    wf: io.TextIOWrapper
    with open(f"./eval/results/{eval_str}_results.csv", "a", newline="") as wf:
        csv_writer = csv.writer(wf)
        if has_id:
            header = ["id", "db_id", "gold_query", "pred_query", "success", "status", "error"]
        else:
            header = ["db_id", "gold_query", "pred_query", "success", "status", "error"]
        csv_writer.writerow(header)


def write_eval_results_row(
    eval_str: str,
    db_id: str,
    gold_query: str,
    success: bool,
    status: str,
    pred_query: str = "",
    error: Optional[str] = None,
    query_id: Optional[str] = None,
):
    wf: io.TextIOWrapper
    with open(f"./eval/results/{eval_str}_results.csv", "a", newline="") as wf:
        csv_writer = csv.writer(wf)

        # Clean the data
        row_data = [db_id, gold_query, pred_query, success, status, error or ""]
        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]

        if query_id:
            row_data.insert(0, query_id)

        csv_writer.writerow(row_data)


def summarize_eval_results(eval_str: str) -> None:
    """
    Reads the results tsv associated with a given eval_str, counts the occurrences of each
    status value, and prints a summary.
    """
    # Read the results tsv into a pandas DataFrame
    results_path = f"./eval/results/{eval_str}_results.csv"
    df = pd.read_csv(results_path)

    # Count occurrences of each status value
    status_counts = Counter(df["status"])
    total_count = len(df)

    # Print summary
    print(f"Summary for evaluation: {eval_str}")
    print(f"Results file: {results_path}")
    for status, count in status_counts.items():
        percentage = (count / total_count) * 100
        print(f"Status: {status} - Count: {count} - Percentage: {percentage:.2f}%")

    print(f"Total entries: {total_count}")
