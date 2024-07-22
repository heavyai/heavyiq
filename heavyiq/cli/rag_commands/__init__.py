import asyncio
from typing import Literal
from uuid import uuid4

import click
from llama_index.core.postprocessor import SimilarityPostprocessor
from pydantic import BaseModel

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyrag.filters import get_facts_filter_matches
from heavyrag.logger import logger
from heavyrag.postprocessor import ReRanker, ReRankerType

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
    from heavyrag.ingest import ainsert_facts

    tasks = []
    for dbname, rows in read_and_group_csv(dataset, group_by_column_idx=1, header=True).items():
        altered_rows = [(row[0], row[2]) for row in rows]  # make list of tuples like (guidance_id, guidance_snippet)
        tasks.append(ainsert_facts(altered_rows, heavydb_name=dbname))
    await asyncio.gather(*tasks)


class RetrieveFactsArgs(BaseModel):
    question: str
    dbname: str
    similarity_top_k: int
    similarity_cutoff: float
    rerank_top_k: int
    rerank_cutoff: float
    reranker_type: ReRankerType


async def retrieve_facts(args: RetrieveFactsArgs) -> list[tuple[str, dict, float]]:
    """
    Retrieve aand re-rank snippets/facts.
    """
    from heavyrag.index import get_index

    index = get_index(collection_name=args.dbname)
    if not index:
        raise ValueError(f"VectorStoreIndex not found {args.dbname} collection.")

    facts_engine = index.as_retriever(
        filters=get_facts_filter_matches(args.dbname),
        similarity_top_k=args.similarity_top_k,
    )
    nodes = await facts_engine.aretrieve(args.question)
    logger.debug(f"Node count after retrieval: {len(nodes)}")
    # apply similarity cutoff
    processor = SimilarityPostprocessor(similarity_cutoff=args.similarity_cutoff)
    filtered_nodes = processor.postprocess_nodes(nodes)
    logger.debug(f"Node count after applying similarity cutoff {args.similarity_cutoff}: {len(filtered_nodes)}")
    # apply re-ranker
    reranker = ReRanker(reranker_type=args.reranker_type, top_n=args.rerank_top_k, cutoff_score=args.rerank_cutoff)
    nodes = await reranker._apostprocess_nodes(nodes=filtered_nodes, query_str=args.question)
    logger.debug(
        f"Node count after applying reranking with top_n and cutoff as {args.rerank_top_k}, {args.rerank_cutoff}: {len(nodes)}"
    )
    return [(node.get_text(), node.metadata, node.score) for node in nodes]


@snippets.command()
@coro
@click.option(
    "--reranker-type", default="hint", show_default=True, help="Type of reranker to use. (e.g., hint, document, tei)"
)
@click.argument("dataset", type=str)
async def eval(dataset: str, reranker_type: Literal["document", "hint", "tei"]):
    """
    Evaluate RAG get_relevant_info function.
    """

    config, rag_eval_id = get_config(), uuid4().hex[:8]
    rag_str = f"rag_{reranker_type}_{rag_eval_id}"
    output_file_path = f"./eval/results/{rag_str}_queries.csv"
    logger.info(f"RAG Eval ID: {rag_eval_id}")
    reranker_tasks, output_rows, final_rows_to_write = (
        [],
        [],
        [],
    )
    if reranker_type not in ["document", "hint", "tei"]:
        raise ValueError(f"Invalid reranker_type {reranker_type}")

    for dbname, rows in read_and_group_csv(dataset, group_by_column_idx=1, header=True).items():
        for row in rows:
            reranker_args = None
            if reranker_type == "document":
                reranker_args = RetrieveFactsArgs(
                    question=row[2],
                    dbname=dbname,
                    similarity_top_k=config.rag_facts_similarity_top_k,
                    similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                    rerank_top_k=config.rag_facts_reranker_top_k,
                    rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                    reranker_type=ReRankerType.DefaultLLMReRanker,
                )
            elif reranker_type == "hint":
                reranker_args = RetrieveFactsArgs(
                    question=row[2],
                    dbname=dbname,
                    similarity_top_k=config.rag_facts_similarity_top_k,
                    similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                    rerank_top_k=config.rag_facts_reranker_top_k,
                    rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                    reranker_type=ReRankerType.OverridedLLMReRanker,
                )
            else:
                reranker_args = RetrieveFactsArgs(
                    question=row[2],
                    dbname=dbname,
                    similarity_top_k=config.rag_facts_similarity_top_k,
                    similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                    rerank_top_k=config.rag_facts_reranker_top_k,
                    rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                    reranker_type=ReRankerType.TEIReRanker,
                )
            reranker_tasks.append(retrieve_facts(reranker_args))
            output_rows.append(row)

    retrieved_contents = await asyncio.gather(*reranker_tasks)

    for output_row, output in zip(output_rows, retrieved_contents):
        predicted_answer = []
        answer = string_to_list(output_row[3])
        also_acceptable_answer = string_to_list(output_row[4])

        for _, snippet_metadata, _ in output:
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
    sum_false_positive, sum_false_negative = sum([i[7] for i in final_rows_to_write]), sum(
        [i[8] for i in final_rows_to_write]
    )
    print(
        f"Total row count: {total_row_count}\n"
        f"Summary for evaluation: {rag_str}\n"
        f"Results file: {output_file_path}\n"
        f"Sum of {reranker_type} false positives: {sum_false_positive}\n"
        f"Sum of {reranker_type} false negatives: {sum_false_negative}\n"
    )
