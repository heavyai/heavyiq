import asyncio
import os
import warnings
from uuid import uuid4

import aiofiles
import click
from aiocsv.readers import AsyncReader
from aiocsv.writers import AsyncWriter
from fastapi.concurrency import run_in_threadpool
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langsmith import Client

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.logging_utils import get_heavyiq_logger

from .utils import (
    awrite_gen_results_header,
    awrite_gen_results_row,
    batched_generator,
    extract_tables_from_query,
    generate_cot,
    generate_cot_chain_input,
    generate_record,
    sql_rate_reply,
    write_cot_output_to_csv,
)

warnings.filterwarnings("ignore")


@click.group()
def gen():
    """Gen data."""
    pass


async def generate_queries(chain: Runnable, chain_input: dict, batch_size: int):
    return await chain.abatch(inputs=[chain_input] * batch_size)


async def process_gen_row_lcel(
    semaphore: asyncio.Semaphore,
    row: tuple | list,
    has_id: bool,
    gen_str: str,
    chain: Runnable,
    max_gens: int,
    max_correct_gens: int,
    include_gold_query: bool = False,
    verbose: bool = True,
):
    logger = get_heavyiq_logger()

    if has_id:
        query_id, db_id, question, gold_query = row
    else:
        query_id = None
        db_id, question, gold_query = row

    async with semaphore:
        correct_gens, gen_idx, num_correct_gens, session_id = (
            [],
            0,
            0,
            await HeavyDB.create_session_id_async(db_name=db_id),
        )
        async with heavydb_context(session_id) as db:
            logger.info(f"Processing Question: {question}")
            tables = await run_in_threadpool(extract_tables_from_query, db._conn, gold_query)
            chain_input = {"question": question, "session_id": session_id, "tables": tables}

            while gen_idx < max_gens and num_correct_gens < max_correct_gens:
                gen_idx += max_correct_gens
                batch_responses = await generate_queries(
                    chain=chain, chain_input=chain_input, batch_size=max_correct_gens
                )
                for response in batch_responses:
                    error, pred_query = response["error"], response["query"]
                    if error:
                        logger.error(f"Failed to generate SQL: {error}")
                    else:
                        # successful response
                        eval_res, query_stats = await asyncio.gather(
                            run_in_threadpool(sql_rate_reply, gold_query, pred_query, db=db),
                            db.aquery_stats(pred_query),
                        )
                        if eval_res["success"] and (pred_query not in correct_gens):
                            correct_gens.append(pred_query)
                            num_correct_gens += 1

        print(correct_gens)
        if include_gold_query:
            await awrite_gen_results_row(
                gen_str,
                "0",
                db_id,
                question,
                gold_query,
                query_id=query_id,
            )
        for idx, gen in enumerate(correct_gens):
            await awrite_gen_results_row(gen_str, str(idx + 1), db_id, question, gen, query_id=query_id)


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
                res = await chain.acall(
                    {chain.input_key: question, "tables": tables}, include_run_info=is_langsmith_active
                )
                pred_query = res[chain.output_key]
                print(pred_query)
                eval_res, query_stats = await asyncio.gather(
                    run_in_threadpool(sql_rate_reply, gold_query, pred_query, db=db), db.aquery_stats(pred_query)
                )
                if eval_res["success"] and pred_query not in correct_gens:
                    correct_gens.append(pred_query)
                    num_correct_gens += 1
                # print(f"Generated query: {pred_query}")
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
                query_id=query_id,
            )
        for idx, gen in enumerate(correct_gens):
            await awrite_gen_results_row(gen_str, str(idx + 1), db_id, question, gen, query_id=query_id)


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
    ctx: click.Context,
    gen_dataset_csv: str,
    temperature: float,
    top_k: int,
    max_gens: int,
    max_correct_gens: int,
    verbose: bool,
) -> None:
    from heavyiq.langchain.utils import is_langsmith_active
    from heavyiq.lcel.chains.heavydb.sql_chain import (
        StrOutputParser,
        final_step,
        query_prompt,
        query_variables,
        validation_step,
    )

    logger = get_heavyiq_logger()

    if not await run_in_threadpool(os.path.exists, gen_dataset_csv):
        raise Exception(f"gen_dataset_csv does not exist: {gen_dataset_csv}")

    gen_id = uuid4().hex[:8]
    gen_str = f"gen_{gen_id}"
    logger.info(f"Gen ID: {gen_id}")

    tasks, semaphore = [], asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=temperature, top_k=top_k)

    chain = (
        RunnablePassthrough().assign(sql_cmd=query_variables | query_prompt | llm | StrOutputParser())
        | validation_step
        | final_step
    )

    # csv must have columns: id (optional), db_id, tables, question, answer
    async with aiofiles.open(gen_dataset_csv, mode="r") as f:
        csv_reader = AsyncReader(f)
        header = await anext(csv_reader)
        has_id = "id" == header[0]  # type: ignore

        await awrite_gen_results_header(gen_str, has_id)

        async for row in csv_reader:
            task = asyncio.create_task(
                process_gen_row_lcel(
                    semaphore,
                    row,
                    has_id=has_id,
                    gen_str=gen_str,
                    chain=chain,
                    verbose=verbose,
                    max_gens=max_gens,
                    max_correct_gens=max_correct_gens,
                )
            )
            tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    results_path = f"./gen/results/{gen_str}_queries.csv"
    print(f"Generated queries: {results_path}")


@gen.command()
@coro
@click.option("--gpt_model", type=str, default="gpt-3.5-turbo-16k", help="OpenAI GPT model to be used.")
@click.option("--batch_size", type=int, default=20, help="Batch size")
@click.argument("gen_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def generate_chain_of_thoughts(ctx: click.Context, gen_dataset_csv: str, batch_size: int, gpt_model: str):
    """
    gen cli command responsible for generating Chain of Thoughts reasoning based on the input question and golden query.
    """
    from .chain.cot_chain import chain

    logger = get_heavyiq_logger()
    if not await run_in_threadpool(os.path.exists, gen_dataset_csv):
        raise Exception(f"gen_dataset_csv does not exist: {gen_dataset_csv}")

    gen_id = uuid4().hex[:8]
    gen_str = f"gen_{gen_id}"
    logger.info(f"Invoking generate_chain_of_thoughts with gen_id: {gen_id}")

    record_generator = batched_generator(generate_record(gen_dataset_csv, gen_str), batch_size=batch_size)
    chain_input_generator = generate_cot_chain_input(record_generator)
    llm_config = {"configurable": {"llm_model": gpt_model}}
    cot_generator = generate_cot(chain_input_generator, chain=chain, config=llm_config)  # type: ignore
    await write_cot_output_to_csv(cot_generator, gen_str=gen_str)
