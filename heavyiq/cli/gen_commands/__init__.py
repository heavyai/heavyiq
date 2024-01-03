import asyncio
import os
from uuid import uuid4

import aiofiles
import click
from aiocsv.readers import AsyncReader
from aiocsv.writers import AsyncWriter
from fastapi.concurrency import run_in_threadpool
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langsmith import Client

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.logging_utils import get_heavyiq_logger

from .utils import (
    awrite_gen_results_header,
    awrite_gen_results_row,
    extract_tables_from_query,
    sql_rate_reply,
)

import warnings
warnings.filterwarnings('ignore')


@click.group()
def gen():
    """Gen data."""
    pass


async def process_gen_row(
    semaphore: asyncio.Semaphore,
    row: tuple | list,
    has_id: bool,
    gen_str: str,
    llm: BaseLLM | BaseChatModel,
    max_gens: int,
    max_correct_gens: int,
    include_gold_query: bool = True,
    verbose: bool = True,
):
    """
    Function to process each row exists on eval dataset.
    """
    from heavyiq.langchain.utils import is_langsmith_active

    logger = get_heavyiq_logger()

    if has_id:
        query_id, db_id, question, gold_query = row
    else:
        query_id = None
        db_id, question, gold_query = row
    async with semaphore:
        logger.info(f"Processing Question: {question}")
        db = await HeavyDB.from_env_async(db_name=db_id)
        tables = extract_tables_from_query(db._conn, gold_query)
        chain = get_nl_to_sql_chain_by_llm(llm)(
            database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[gen_str, "cli"]
        )

        correct_gens = []
        gen_idx = 0
        num_correct_gens = 0
        
        while gen_idx < max_gens and num_correct_gens < max_correct_gens: 
            query_stats = None
            try:
                gen_idx += 1
                res = await chain.acall({chain.input_key: question, "tables": tables}, include_run_info=is_langsmith_active)
                pred_query = res[chain.output_key]
                print(pred_query)
                eval_res, query_stats = await asyncio.gather(
                    run_in_threadpool(sql_rate_reply, gold_query, pred_query, db=db), db.aquery_stats(pred_query)
                )
                if eval_res["success"] and pred_query not in correct_gens:
                    correct_gens.append(pred_query)
                    num_correct_gens += 1
                #print(f"Generated query: {pred_query}")
            except NLtoSQLException as e:
                logger.exception(f"Failed to generate SQL: {e}")
            except Exception as e:
                logger.exception(f"Failed to generate SQL: {e}")
        print(correct_gens)
        if include_gold_query:
            await awrite_gen_results_row(
                gen_str,
                "0",
                db_id,
                question,
                gold_query,
                query_id = query_id,
            )
        for idx, gen in enumerate(correct_gens):
            await awrite_gen_results_row(
                gen_str,
                str(idx + 1),
                db_id,
                question,
                gen,
                query_id = query_id
            )



@gen.command()
@coro
@click.option("--temperature", default=0.7, help="Temperature for LLM (Defaults to 0.7)", type=float)
@click.option("--top_k", default=40, help="Top-K for LLM (Defaults to 40)", type=int)
@click.option("--max_gens", default=10, help="Maximum number of generations", type=int)
@click.option("--max_correct_gens", default=3, help="Maximum number of correct generations", type=int)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("gen_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_questions(
    ctx: click.Context, gen_dataset_csv: str, temperature: float, top_k: int, max_gens: int, max_correct_gens: int, verbose: bool
) -> None:
    from heavyiq.langchain.utils import is_langsmith_active

    logger = get_heavyiq_logger()

    if not await run_in_threadpool(os.path.exists, gen_dataset_csv):
        raise Exception(f"gen_dataset_csv does not exist: {gen_dataset_csv}")

    gen_id = uuid4().hex[:8]
    gen_str = f"gen_{gen_id}"
    logger.info(f"Gen ID: {gen_id}")

    tasks, semaphore = [], asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=temperature, top_k=top_k)

    config = get_config()
    # csv must have columns: id (optional), db_id, tables, question, answer
    async with aiofiles.open(gen_dataset_csv, mode="r") as f:
        csv_reader = AsyncReader(f)
        header = await anext(csv_reader)
        has_id = "id" == header[0]  # type: ignore

        await awrite_gen_results_header(gen_str, has_id)

        async for row in csv_reader:
            task = asyncio.create_task(
                process_gen_row(
                    semaphore,
                    row,
                    has_id=has_id,
                    gen_str=gen_str,
                    llm=llm,
                    verbose=verbose,
                    max_gens=max_gens,
                    max_correct_gens=max_correct_gens,
                )
            )
            tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    results_path = f"./gen/results/{gen_str}_queries.csv"
    print(f"Generated queries: {results_path}")
