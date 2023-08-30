from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time

import click
from langchain.callbacks import get_openai_callback
from langchain.base_language import BaseLanguageModel

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_llm_by_model_name
from heavyiq.utils import strip_sql_comments


@click.group()
def eval():
    """Evaluate models."""
    pass


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
    runner_llm = get_llm_by_model_name(model, temperature=temperature, client=None)
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
