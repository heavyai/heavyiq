from uuid import uuid4
import os
import io
import csv

import click
from langsmith import Client

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_llm_by_type, LLMType

from .utils import sql_rate_reply, write_eval_results_header, write_eval_results_row, summarize_eval_results


@click.group()
def eval():
    """Evaluate models."""
    pass


@eval.command()
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("eval_dataset_csv", type=str)
@click.pass_context
def run_config_model_on_questions(ctx: click.Context, eval_dataset_csv: str, temperature: float, verbose: bool) -> None:
    """Call the NL to SQL Chain on each question in eval_questions.csv using LLM from config file"""
    from heavyiq.langchain.utils import is_langsmith_active

    if not os.path.exists(eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    print(f"Eval ID: {eval_id}")

    # csv must have columns: id (optional), db_id, tables, question, answer
    f: io.TextIOWrapper
    with open(eval_dataset_csv, mode="r") as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        has_id = "id" == header[0]

        write_eval_results_header(eval_str, has_id)

        for row in csv_reader:
            if has_id:
                query_id, db_id, tables, question, gold_query = row
            else:
                query_id = None
                db_id, tables, question, gold_query = row
            tables: list[str] = [table.strip("'") for table in str(tables).split(",")]
            print(f"Processing Question: {question}")
            llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=temperature)
            db = HeavyDB.from_env(db_name=db_id, include_tables=tables)
            chain = get_nl_to_sql_chain_by_llm(llm)(
                database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[eval_str, "cli"]
            )
            try:
                res = chain({chain.input_key: question}, include_run_info=is_langsmith_active)
                pred_query = res[chain.output_key]
                print(f"Generated SQL: {pred_query}")
                print("Evaluating SQL")
                eval_res = sql_rate_reply(db_id, gold_query, pred_query)
                print(f"Evaluation Success: {eval_res['success']}")
                print(f"Evaluation Status: {eval_res['status']}")
                if is_langsmith_active:
                    feedback_id = str(res["__run"].run_id)
                    print(f"Langsmith Run ID: {feedback_id}")
                    langsmith_client = Client()
                    langsmith_client.create_feedback(
                        feedback_id, "eval_status", score=eval_res["success"], comment=eval_res["status"]
                    )
                print("=====================================")
                write_eval_results_row(
                    eval_str,
                    db_id,
                    gold_query,
                    eval_res["success"],
                    eval_res["status"],
                    pred_query,
                    query_id=query_id,
                    error=eval_res["error"],
                )
            except NLtoSQLException as e:
                print(e)
                print("Failed to generate SQL")
                write_eval_results_row(
                    eval_str,
                    db_id,
                    gold_query,
                    False,
                    "failed_to_generate_sql",
                    e.failed_sql,
                    query_id=query_id,
                )
            except Exception as e:
                print(e)
                print("Failed to generate SQL")

    summarize_eval_results(eval_str)
