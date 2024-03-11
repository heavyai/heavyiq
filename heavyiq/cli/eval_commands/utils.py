import asyncio
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
from fastapi.concurrency import run_in_threadpool

from heavyiq.langchain import HeavyDB


def compute_prob_stats(selected_tokens: list[str], top_log_probs: list[dict[str, float]]) -> dict[str, Any]:
    selected_tokens_and_probs = []
    sum_log_probs = 0.0
    min_prob = 1.0
    min_prob_token = None
    prob_decile_histogram = [0] * 10

    num_calc_tokens = 0
    select_seen = False
    for selected_token, token_logprobs in zip(selected_tokens, top_log_probs):
        selected_token_log_prob = token_logprobs[selected_token]
        # Convert the log probability to actual probability
        selected_token_prob = math.exp(selected_token_log_prob)
        # We don't use the probabilities on any tokens up to and including the first SELECT, as depending on the
        # prompt the LLM will sometimes want to put a newline token first, so using these first probabilities
        # will artifically lower the avg, total, and min probabilities

        # Todo (Todd): Consider the case where we have a query starting with a WITH (CTE)

        if select_seen:
            num_calc_tokens += 1
            sum_log_probs += selected_token_log_prob
            hist_bin = 9 if selected_token_prob >= 1.0 else int(selected_token_prob * 10)
            prob_decile_histogram[hist_bin] += 1
            if selected_token_prob < min_prob:
                min_prob = selected_token_prob
                min_prob_token = selected_token
        else:
            if selected_token == "▁SELECT" or selected_token == "SELECT":
                select_seen = True

        # Append the result to the list
        selected_tokens_and_probs.append({"token": selected_token, "probability": selected_token_prob})

    prob_stats = {}
    prob_stats["num_tokens"] = len(selected_tokens)
    prob_stats["total_prob"] = math.exp(sum_log_probs)
    prob_stats["avg_prob"] = math.exp(sum_log_probs / num_calc_tokens)
    prob_stats["min_prob"] = min_prob
    prob_stats["min_prob_token"] = min_prob_token
    prob_stats["prob_decile_histogram"] = prob_decile_histogram
    prob_stats["selected_token_probs"] = selected_tokens_and_probs

    return prob_stats


async def sql_rate_reply(
    gold_query: str, pred_query: str, db_id: str | None = None, db: HeavyDB | None = None, question: str | None = None
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
            db = await HeavyDB.from_env_async(db_name=db_id)
        gold_df_awaitable = run_in_threadpool(pd.read_sql, gold_query, db._conn)
        pred_df_awaitable = run_in_threadpool(pd.read_sql, pred_query, db._conn)
        gold_df, pred_df = await gold_df_awaitable, await pred_df_awaitable
        num_gold_rows = len(gold_df.axes[0])  # type: ignore
        num_pred_rows = len(pred_df.axes[0])  # type: ignore
        num_gold_cols = len(gold_df.axes[1])  # type: ignore
        num_pred_cols = len(pred_df.axes[1])  # type: ignore
        if num_gold_rows != num_pred_rows:
            print("ROW COUNT MISMATCH")
            print(question)
            print(gold_query)
            print(pred_query)
            query_metadata["status"] = "row_count_mismatch"
            return query_metadata
        if num_gold_cols != num_pred_cols:
            print("COL COUNT MISMATCH")
            print(question)
            print(gold_query)
            print(pred_query)
            query_metadata["status"] = "col_count_mismatch"
            return query_metadata
        gold_query_has_order_by = gold_query.lower().find("order by") >= 0
        dfs_are_equal = None
        if gold_query_has_order_by:
            dfs_are_equal = await run_in_threadpool(np.array_equal, gold_df.values, pred_df.values)
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
                    dfs_are_equal = await run_in_threadpool(np.array_equal, gold_df.values, pred_df_perm.values)
                else:
                    gold_df_sorted = gold_df.sort_values(by=list(gold_df.columns)).reset_index(drop=True)
                    pred_df_perm_sorted = pred_df_perm.sort_values(by=list(pred_df_perm.columns)).reset_index(drop=True)
                    dfs_are_equal = await run_in_threadpool(
                        np.array_equal, gold_df_sorted.values, pred_df_perm_sorted.values
                    )
                if dfs_are_equal:
                    print("COLUMN ORDER DIFF")
                    print(question)
                    query_metadata["success"] = True
                    query_metadata["status"] = "column_order_difference"
                    return query_metadata
            print("VALUES MISMATCH")
            print(question)
            query_metadata["status"] = "values_mismatch"
            print(gold_query)
            print(pred_query)
            return query_metadata
    except Exception as e:
        print("QUERY FAIL")
        query_metadata["status"] = "execution_error"
        query_metadata["error"] = e
        print(question)
        print(gold_query)
        print(pred_query)
        print(e)
        return query_metadata


async def awrite_eval_results_header(
    eval_str: str, has_id: bool, enable_logprobs: bool, enable_query_stats: bool = True
):
    async with aiofiles.open(f"./eval/results/{eval_str}_results.csv", "a", newline="") as wf:
        header = []
        if has_id:
            header.append("query_id")
        header.extend(["db_id", "question", "gold_query", "pred_query", "success", "status", "error"])
        if enable_logprobs:
            header.extend(["num_tokens", "total_prob", "avg_prob", "min_prob", "prob_decile_histogram"])
        if enable_query_stats:
            header.extend(["num_joins", "num_unions", "num_aggs", "num_filters", "num_sorts"])

        writer = AsyncWriter(wf, dialect="unix")
        await writer.writerow(header)


async def awrite_eval_results_row(
    eval_str: str,
    db_id: str,
    question: str,
    gold_query: str,
    success: bool,
    status: str,
    pred_query: str = "",
    error: Optional[str] = None,
    query_id: Optional[str] = None,
    prob_stats: Optional[dict] = None,
    query_stats: Optional[dict] = None,
    enable_logprobs: bool = True,
    enable_querystats: bool = True,
):
    async with aiofiles.open(f"./eval/results/{eval_str}_results.csv", "a", newline="") as wf:
        writer = AsyncWriter(wf, dialect="unix")
        # Clean the data
        row_data = []
        if query_id:
            row_data.append(query_id)
        row_data.extend([db_id, question, gold_query, pred_query, success, status, error or ""])
        if enable_logprobs and prob_stats:
            row_data.extend(
                [
                    prob_stats["num_tokens"],
                    prob_stats["total_prob"],
                    prob_stats["avg_prob"],
                    prob_stats["min_prob"],
                    prob_stats["prob_decile_histogram"],
                ]
            )
        elif enable_logprobs:
            row_data.extend([None, None, None, None, None])
        if enable_querystats and query_stats:
            row_data.extend(
                [
                    query_stats["joins"],
                    query_stats["unions"],
                    query_stats["aggs"],
                    query_stats["filters"],
                    query_stats["sorts"],
                ]
            )
        elif enable_querystats:
            row_data.extend([None, None, None, None, None])

        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]

        await writer.writerow(row_data)


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

    # Compute success metrics
    success_count = df["success"].sum()  # Summing boolean column gives count of True values
    success_percentage = (success_count / total_count) * 100

    # Print summary
    print(f"Summary for evaluation: {eval_str}")
    print(f"Results file: {results_path}")
    for status, count in status_counts.items():
        percentage = (count / total_count) * 100
        print(f"Status: {status} - Count: {count} - Percentage: {percentage:.2f}%")

    print(f"Total entries: {total_count}")
    print(f"Total success: {success_count} - Success Percentage: {success_percentage:.2f}%")


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


async def aextract_tables_from_query(db: HeavyDB, query: str) -> list[str]:
    """
    Extract table names from SQL query async.
    """
    query_plan = await db.aget_calcite_query_plan(query)
    pattern = r"LogicalTableScan\(table=\[\[(.*?)\]\]\)"
    matches = re.findall(pattern, query_plan)
    tables = []
    for match in matches:
        table = match.split(", ")[1]
        tables.append(table)
    return list(set(tables))  # remove duplicates
