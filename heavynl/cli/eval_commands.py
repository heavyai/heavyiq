from uuid import uuid4

import click
from langchain.callbacks import get_openai_callback

from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import (
    NLtoSQLChain,
)
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.utils import strip_sql_comments


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
@click.pass_context
def run_model_on_questions(ctx: click.Context, model: str, temperature: float, verbose: bool, n: int) -> None:
    """Call the NL to SQL Chain on each question in eval_questions.tsv. Provides tables to LLM"""
    # generate UUID for this run, just need the first few digits
    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    print(f"Eval ID: {eval_id}")
    with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
        f.write(
            "eval_id\tmodel\ttemperature\tprimary_table\tis_multi_table\tsecondary_table\tquestion\toutput_sql\ttotal_tokens\tprompt_tokens\tcompletion_tokens\tsuccessful_requests\ttotal_cost\treference_sql\n"
        )

    llm = get_llm_by_model_name(
        model, [eval_str, "cli", "chain", "nl_to_sql_chain"], temperature=temperature, client=None
    )

    with open("./eval/questions.tsv") as f:
        f.readline()  # skip the header
        questions = f.readlines()

        for i in range(n):
            print(f"=========== Run {i+1} of {n} ===========")
            num_of_questions = len(questions)
            num_of_successes = 0
            for index, question in enumerate(questions):
                if index % 5 == 0:
                    print(f"Question {index+1} of {num_of_questions}...")
                primary_table, is_multi_table, secondary_table, question, reference_sql = question.split("\t")
                tables = [primary_table]
                if is_multi_table == "True":
                    tables.append(secondary_table)
                heavydb = HeavyDB.from_env(include_tables=tables)
                chain = NLtoSQLChain(database=heavydb, llm=llm, verbose=verbose)
                with get_openai_callback() as cb:
                    try:
                        sql = strip_sql_comments(chain.run(question)).replace("\n", " ")
                        num_of_successes += 1
                    except Exception as e:
                        sql = f"ERROR: {e}".replace("\n", " ")

                    with open(f"./eval/results/{eval_str}_results.tsv", "a") as f:
                        f.write(
                            f"{eval_id}\t{model}\t{temperature}\t{primary_table}\t{is_multi_table}\t{secondary_table}\t{question.strip()}\t{sql}\t{cb.total_tokens}\t{cb.prompt_tokens}\t{cb.completion_tokens}\t{cb.successful_requests}\t{round(cb.total_cost, 4)}\t{reference_sql.strip()}\n"
                        )

            print(
                f"Generated SQL for {num_of_successes}/{num_of_questions} questions. ({round((num_of_successes/num_of_questions)*100, 2)}%)"
            )
