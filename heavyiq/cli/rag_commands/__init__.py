import asyncio
from uuid import uuid4

import click

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyrag.ingest import ainsert_facts
from heavyrag.logger import logger
from heavyrag.main import ask_facts

from .utils import read_and_group_csv, read_csv_as_lists, string_to_list, write_csv


@click.group()
def rag():
    """RAG cli commands."""
    pass


@rag.group()
def snippets():
    """
    RAG snippets command
    """
    pass


@snippets.command()
@coro
@click.argument("dataset", type=str)
async def insert(dataset: str):
    """
    Insert snippets into sqlite db and chroma vectorstore.
    """
    tasks = []
    for dbname, rows in read_and_group_csv(dataset, group_by_column_idx=1, header=True).items():
        altered_rows = [(row[0], row[2]) for row in rows]  # make list of tuples like (guidance_id, guidance_snippet)
        tasks.append(ainsert_facts(altered_rows, heavydb_name=dbname))
    await asyncio.gather(*tasks)


@snippets.command()
@coro
@click.argument("dataset", type=str)
async def eval(dataset: str):
    """
    Evaluate RAG get_relevant_info function.
    """

    config, rag_eval_id = get_config(), uuid4().hex[:8]
    rag_str = f"rag_{rag_eval_id}"
    output_file_path = f"./eval/results/{rag_str}_queries.csv"
    logger.info(f"RAG Eval ID: {rag_eval_id}")
    tasks, output_rows, final_rows_to_write = [], [], []
    for dbname, rows in read_and_group_csv(dataset, group_by_column_idx=1, header=True).items():
        for row in rows:
            tasks.append(
                ask_facts(
                    row[2],
                    dbname,
                    only_retrieve=True,
                    do_evaluate=False,
                    similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                    similarity_top_k=config.rag_facts_similarity_top_k,
                    with_reranker=True,
                )
            )
            output_rows.append(row)

    retrieved_contents = await asyncio.gather(*tasks)
    for output_row, rag_output in zip(output_rows, retrieved_contents):
        predicted_answer = []
        answer = string_to_list(output_row[3])
        also_acceptable_answer = string_to_list(output_row[4])

        for _, snippet_metadata, _ in rag_output:
            predicted_answer.append(int(snippet_metadata["id"]))

        valid_answers = set(answer + also_acceptable_answer)
        false_positive = len(set(predicted_answer) - valid_answers)
        false_negative = len(set(answer) - set(predicted_answer))

        # check for any one predicted answer exists in answer and also_accepted_answer
        # if yes then set status as True else False
        is_success = False
        for i in predicted_answer:
            if i in valid_answers:
                is_success = True
                break

        final_rows_to_write.append(
            output_row[:3]
            + [answer, also_acceptable_answer, predicted_answer, is_success, false_positive, false_negative]
        )

    if final_rows_to_write:
        logger.info("Started writing output csv...")
        write_csv(
            output_file_path,
            headers=[
                "eval_id",
                "db_id",
                "question",
                "answer",
                "also_acceptable_answer",
                "predicted_answer",
                "success",
                "false_positive",
                "false_negative",
            ],
            rows=final_rows_to_write,
        )
        logger.info(f"Successfully written to {output_file_path}")

    total_row_count = len(final_rows_to_write)
    sum_false_positive, sum_false_negative = sum([i[-2] for i in final_rows_to_write]), sum(
        [i[-1] for i in final_rows_to_write]
    )
    print(
        f"Total row count: {total_row_count}\n"
        f"Summary for evaluation: {rag_str}\n"
        f"Results file: {output_file_path}\n"
        f"Sum of false positives: {sum_false_positive}\n"
        f"Sum of false negatives: {sum_false_negative}"
    )
