from uuid import uuid4
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import click
from langchain.callbacks import get_openai_callback
from promptwatch import PromptWatch

from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import (
    NLtoSQLChain,
)
from heavynl.config import get_config
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.utils import strip_sql_comments


@click.group()
def eval():
    """Evaluate models."""
    pass


def promptwatch_context(project: str, tenant: str) -> PromptWatch | nullcontext:
    config = get_config()
    if config.promptwatch_api_key and config.promptlayer_api_key != "":
        return PromptWatch(api_key=config.promptwatch_api_key, tracking_project=project, tracking_tenant=tenant)
    else:
        return nullcontext()


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

    def process_question(heavydb: HeavyDB, index: int, question: str) -> int:
        primary_table, is_multi_table, secondary_table, question, reference_sql = question.split("\t")
        tables = [primary_table]
        if is_multi_table.upper() == "TRUE":
            tables.append(secondary_table)
        chain = NLtoSQLChain(database=heavydb, llm=llm, verbose=verbose)
        with promptwatch_context("nl_to_sql_eval", str(eval_id)), get_openai_callback() as cb:
            try:
                res = chain({"query": question, "tables": tables})
                sql = strip_sql_comments(res["sql"]).replace("\n", " ")
                num_of_successes = 1
            except Exception as e:
                sql = f"ERROR: {e}".replace("\n", " ")
                num_of_successes = 0

            with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
                f.write(
                    f"{eval_id}\t{model}\t{temperature}\t{primary_table}\t{is_multi_table}\t{secondary_table}\t{question.strip()}\t{sql}\t{cb.total_tokens}\t{cb.prompt_tokens}\t{cb.completion_tokens}\t{cb.successful_requests}\t{round(cb.total_cost, 4)}\t{reference_sql.strip()}\n"
                )
        return num_of_successes

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    print(f"Eval ID: {eval_id}")
    with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
        f.write(
            "eval_id\tmodel\ttemperature\tprimary_table\tis_multi_table\tsecondary_table\tquestion\toutput_sql\ttotal_tokens\tprompt_tokens\tcompletion_tokens\tsuccessful_requests\ttotal_cost\treference_sql\n"
        )

    heavydb = HeavyDB.from_env()
    llm = get_llm_by_model_name(
        model, [eval_str, "cli", "chain", "nl_to_sql_chain"], temperature=temperature, client=None
    )
    process_func = partial(process_question, heavydb)

    with open("./eval/questions.tsv") as f:
        f.readline()  # skip the header
        questions = f.readlines()

        for i in range(n):
            print(f"=========== Run {i+1} of {n} ===========")
            num_of_questions = len(questions)
            total_num_of_successes = 0
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_results = {
                    executor.submit(process_func, index, question): question for index, question in enumerate(questions)
                }
                for future in as_completed(future_results):
                    num_of_successes = future.result()
                    total_num_of_successes += num_of_successes
            print(
                f"Generated SQL for {total_num_of_successes}/{num_of_questions} questions. ({round((total_num_of_successes/num_of_questions)*100, 2)}%)"
            )
