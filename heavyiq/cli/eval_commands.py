from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
import os
import io
from typing import Any, Optional
from itertools import permutations

import click
from langchain.callbacks import get_openai_callback
from langchain.base_language import BaseLanguageModel
import pandas as pd
import numpy as np

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_openai_llm_by_model_name, get_llm_by_type, LLMType
from heavyiq.utils import strip_sql_comments


def sql_rate_reply(
    question: str, db_id: str, query_id: Optional[str], gold_query: str, pred_query: str
) -> dict[str, Any]:
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/pred
    query_metadata = {
        "db_id": db_id,
        "query_id": query_id,
        "gold_query": gold_query,
        "pred_query": pred_query,
        "success": False,
        "status": "success",
        "error": None,
    }
    try:
        db = HeavyDB.from_env(db_name=db_id)
        gold_df = pd.read_sql(gold_query, db._conn)
        pred_df = pd.read_sql(pred_query, db._conn)
        num_gold_rows = len(gold_df.axes[0])
        num_pred_rows = len(pred_df.axes[0])
        num_gold_cols = len(gold_df.axes[1])
        num_pred_cols = len(pred_df.axes[1])

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


@click.group()
def eval():
    """Evaluate models."""
    pass


@eval.command()
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("eval_dataset_tsv", type=str)
@click.pass_context
def run_config_model_on_questions(ctx: click.Context, eval_dataset_tsv: str, temperature: float, verbose: bool) -> None:
    """Call the NL to SQL Chain on each question in eval_questions.tsv using LLM from config file"""
    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    print(f"Eval ID: {eval_id}")

    if not os.path.exists(eval_dataset_tsv):
        raise Exception(f"eval_dataset_tsv does not exist: {eval_dataset_tsv}")

    # tsv must have columns: id (optional), db_id, tables, question, answer
    f: io.TextIOWrapper
    with open(eval_dataset_tsv) as f:
        header = f.readline().strip()
        has_id = "id" == header.split("\t")[0]

        for line in f.readlines():
            if has_id:
                id, db_id, tables, question, answer = line.strip().split("\t")
            else:
                id = None
                db_id, tables, question, answer = line.strip().split("\t")
            tables: list[str] = [table.strip("'") for table in str(tables).split(",")]
            print(f"Processing Question: {question}")
            llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=temperature)
            db = HeavyDB.from_env(db_name=db_id, include_tables=tables)
            chain = get_nl_to_sql_chain_by_llm(llm)(
                database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[eval_str, "cli"]
            )
            try:
                res = chain({chain.input_key: question})[chain.output_key]
                print(f"Generated SQL: {res}")
            except Exception:
                print("Failed to generate SQL")
                res = None
            if res is not None:
                print("Evaluating SQL")
                eval_res = sql_rate_reply(question, db_id, id, answer, res)
                print(f"Evaluation Status: {eval_res['status']}")


@eval.command()
@click.option(
    "--model",
    default="text-davinci-003",
    help="Specify the model to use for the LLM. (Defaults to 'text-davinci-003')",
)
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.option("--n", default=1, help="How many times to run model on each question", type=int)
@click.option("--max_threads", default=8, help="Maximum number of threads. (Defaults to 8)", type=int)
@click.pass_context
def run_model_on_questions(
    ctx: click.Context, model: str, temperature: float, verbose: bool, n: int, max_threads: int
) -> None:
    """Call the NL to SQL Chain on each question in eval_questions.tsv. Provides tables to LLM"""

    def process_question(
        eval_str: str, heavydb: HeavyDB, llm: BaseLanguageModel, index: int, question: str
    ) -> tuple[int, float]:
        start_time = time.time()
        primary_table, is_multi_table, secondary_table, question, reference_sql = question.split("\t")
        tables = [primary_table]
        if is_multi_table.upper() == "TRUE":
            tables.append(secondary_table)
        chain = get_nl_to_sql_chain_by_llm(llm)(database=heavydb, llm=llm, verbose=verbose, tags=[eval_str, "cli"])  # type: ignore
        with get_openai_callback() as cb:
            try:
                res = chain({"query": question, "tables": tables})
                sql = strip_sql_comments(res["sql"]).replace("\n", " ")
                num_of_successes = 1
            except Exception as e:
                sql = f"ERROR: {e}".replace("\n", " ")
                num_of_successes = 0
            elapsed_time = time.time() - start_time

            with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
                f.write(
                    f"{eval_id}\t{model}\t{temperature}\t{primary_table}\t{is_multi_table}\t{secondary_table}\t{question.strip()}\t{sql}\t{cb.total_tokens}\t{cb.prompt_tokens}\t{cb.completion_tokens}\t{cb.successful_requests}\t{round(cb.total_cost, 4)}\t{reference_sql.strip()}\n"
                )

        return num_of_successes, elapsed_time

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    print(f"Eval ID: {eval_id}")
    with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
        f.write(
            "eval_id\tmodel\ttemperature\tprimary_table\tis_multi_table\tsecondary_table\tquestion\toutput_sql\ttotal_tokens\tprompt_tokens\tcompletion_tokens\tsuccessful_requests\ttotal_cost\treference_sql\n"
        )

    heavydb = HeavyDB.from_env()
    runner_llm = get_openai_llm_by_model_name(model, temperature=temperature, client=None)
    process_func = partial(process_question, eval_str, heavydb, runner_llm)

    with open("./eval/questions.tsv") as f:
        f.readline()  # skip the header
        questions = f.readlines()

        for i in range(n):
            print(f"=========== Run {i+1} of {n} ===========")
            num_of_questions = len(questions)
            total_num_of_successes = 0
            total_time = 0.0
            min_time = float("inf")
            max_time = 0.0

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_results = {
                    executor.submit(process_func, index, question): question for index, question in enumerate(questions)
                }
                for future in as_completed(future_results):
                    num_of_successes, elapsed_time = future.result()
                    total_time += elapsed_time
                    min_time = min(min_time, elapsed_time)
                    max_time = max(max_time, elapsed_time)
                    total_num_of_successes += num_of_successes
            print(
                f"Generated SQL for {total_num_of_successes}/{num_of_questions} questions. ({round((total_num_of_successes/num_of_questions)*100, 2)}%)"
            )
            print(f"Total time: {round(total_time, 2)}s")
            print(f"Min time: {round(min_time, 2)}s")
            print(f"Max time: {round(max_time, 2)}s")
            print(f"Avg time: {round(total_time/num_of_questions, 2)}s")


@eval.command()
@click.pass_context
def run_complexity_rater(ctx: click.Context) -> None:
    """Call the rate_sql_complexity utility on each reference SQL in eval/questions.tsv."""

    heavydb = HeavyDB.from_env()

    from heavyiq.utils import rate_sql_complexity

    with open("./eval/questions.tsv") as f:
        f.readline()  # skip the header
        questions = f.readlines()

        for line in questions:
            primary_table, is_multi_table, secondary_table, question, reference_sql = line.split("\t")
            print(f"Reference SQL: {reference_sql}\n")
            plan = heavydb.get_query_plan(reference_sql)
            print(f"Plan: {plan}\n")
            print(f"Complexity: {rate_sql_complexity(plan)}\n")
            print("====================================")
