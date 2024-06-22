import asyncio
from uuid import uuid4

import click
from llama_index.core.postprocessor import SimilarityPostprocessor
from pydantic import BaseModel

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyrag.filters import get_facts_filter_matches
from heavyrag.index import get_index
from heavyrag.ingest import ainsert_facts
from heavyrag.logger import logger
from heavyrag.main import ask_facts
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
@click.argument("dataset", type=str)
async def eval(dataset: str):
    """
    Evaluate RAG get_relevant_info function.
    """

    config, rag_eval_id = get_config(), uuid4().hex[:8]
    rag_str = f"rag_{rag_eval_id}"
    output_file_path = f"./eval/results/{rag_str}_queries.csv"
    logger.info(f"RAG Eval ID: {rag_eval_id}")
    deafult_llm_reranker_tasks, hint_llm_reranker_tasks, tei_reranker_tasks, output_rows, final_rows_to_write = (
        [],
        [],
        [],
        [],
        [],
    )
    for dbname, rows in read_and_group_csv(dataset, group_by_column_idx=1, header=True).items():
        for row in rows:
            default_args = RetrieveFactsArgs(
                question=row[2],
                dbname=dbname,
                similarity_top_k=config.rag_facts_similarity_top_k,
                similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                rerank_top_k=config.rag_facts_reranker_top_k,
                rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                reranker_type=ReRankerType.DefaultLLMReRanker,
            )
            hint_args = RetrieveFactsArgs(
                question=row[2],
                dbname=dbname,
                similarity_top_k=config.rag_facts_similarity_top_k,
                similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                rerank_top_k=config.rag_facts_reranker_top_k,
                rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                reranker_type=ReRankerType.OverridedLLMReRanker,
            )
            tei_args = RetrieveFactsArgs(
                question=row[2],
                dbname=dbname,
                similarity_top_k=config.rag_facts_similarity_top_k,
                similarity_cutoff=config.rag_facts_similarity_cutoff_score,
                rerank_top_k=config.rag_facts_reranker_top_k,
                rerank_cutoff=config.rag_facts_reranker_cutoff_score,
                reranker_type=ReRankerType.TEIReRanker,
            )
            deafult_llm_reranker_tasks.append(retrieve_facts(default_args))
            hint_llm_reranker_tasks.append(retrieve_facts(hint_args))
            tei_reranker_tasks.append(retrieve_facts(tei_args))
            output_rows.append(row)

    retrieved_contents_from_default_prompt = await asyncio.gather(*deafult_llm_reranker_tasks)
    retrieved_contents_from_hint_prompt = await asyncio.gather(*hint_llm_reranker_tasks)
    retrieved_contents_from_tei = await asyncio.gather(*tei_reranker_tasks)

    for output_row, hint_output, default_output, tei_output in zip(
        output_rows,
        retrieved_contents_from_hint_prompt,
        retrieved_contents_from_default_prompt,
        retrieved_contents_from_tei,
    ):
        predicted_answer, predicted_answer_default, predicted_answer_tei = [], [], []
        answer = string_to_list(output_row[3])
        also_acceptable_answer = string_to_list(output_row[4])

        for _, snippet_metadata, _ in hint_output:
            predicted_answer.append(int(snippet_metadata["id"]))
        for _, snippet_metadata, _ in default_output:
            predicted_answer_default.append(int(snippet_metadata["id"]))
        for _, snippet_metadata, _ in tei_output:
            predicted_answer_tei.append(int(snippet_metadata["id"]))

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
            + [
                answer,
                also_acceptable_answer,
                predicted_answer,
                is_success,
                false_positive,
                false_negative,
                predicted_answer_default,
                predicted_answer_tei,
            ]
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
                "default_predicted_answer",
                "tei_predicted_answer",
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
        f"Sum of false positives: {sum_false_positive}\n"
        f"Sum of false negatives: {sum_false_negative}"
    )
