from uuid import uuid4
import os
import asyncio
import aiofiles
from aiofiles.ospath import exists as aexists
from aiocsv.readers import AsyncReader
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel

import click
from langsmith import Client
from fastapi.concurrency import run_in_threadpool
from heavyiq.cli.decorators import coro
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_llm_by_type, LLMType
from heavyiq.logging_utils import get_heavyiq_logger

from .utils import (
    sql_rate_reply,
    summarize_eval_results,
    awrite_eval_results_header,
    awrite_eval_results_row,
)


@click.group()
def eval():
    """Evaluate models."""
    pass


async def process_eval_row(
    semaphore: asyncio.Semaphore,
    row: tuple | list,
    has_id: bool,
    eval_str: str,
    llm: BaseLLM | BaseChatModel,
    verbose: bool = True,
):
    """
    Function to process each row exists on eval dataset.
    """
    from heavyiq.langchain.utils import is_langsmith_active

    logger = get_heavyiq_logger()

    if has_id:
        query_id, db_id, tables, question, gold_query = row
    else:
        query_id = None
        db_id, tables, question, gold_query = row

    async with semaphore:
        tables = [table.strip("'") for table in str(tables).split(",")]
        logger.info(f"Processing Question: {question}")
        db = await HeavyDB.from_env_async(db_name=db_id, include_tables=tables)
        chain = get_nl_to_sql_chain_by_llm(llm)(
            database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[eval_str, "cli"]
        )

        try:
            res = await chain.acall({chain.input_key: question}, include_run_info=is_langsmith_active)
            pred_query = res[chain.output_key]
            logger.info(f"Generated SQL: {pred_query}")
            logger.debug("Evaluating SQL")
            eval_res = await run_in_threadpool(sql_rate_reply, db_id, gold_query, pred_query)
            logger.info(f"Evaluation Success: {eval_res['success']}")
            logger.debug(f"Evaluation Status: {eval_res['status']}")
            if is_langsmith_active:
                feedback_id = str(res["__run"].run_id)
                logger.debug(f"Langsmith Run ID: {feedback_id}")
                langsmith_client = Client()
                await run_in_threadpool(
                    langsmith_client.create_feedback,
                    feedback_id,
                    "eval_status",
                    score=eval_res["success"],
                    comment=eval_res["status"],
                )
            logger.debug("=====================================")
            await awrite_eval_results_row(
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
            logger.exception(f"Failed to generate SQL: {e}")
            await awrite_eval_results_row(
                eval_str,
                db_id,
                gold_query,
                False,
                "failed_to_generate_sql",
                e.failed_sql,  # type: ignore
                query_id=query_id,
            )
        except Exception as e:
            logger.exception(f"Failed to generate SQL: {e}")


@eval.command()
@coro
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("eval_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_questions(
    ctx: click.Context, eval_dataset_csv: str, temperature: float, verbose: bool
) -> None:
    from heavyiq.langchain.utils import is_langsmith_active

    logger = get_heavyiq_logger()

    if not await aexists(eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Eval ID: {eval_id}")

    tasks, semaphore = [], asyncio.Semaphore(3)  # Limit to 3 concurrent tasks

    llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=temperature)

    # csv must have columns: id (optional), db_id, tables, question, answer
    async with aiofiles.open(eval_dataset_csv, mode="r") as f:
        csv_reader = AsyncReader(f)
        header = await anext(csv_reader)
        has_id = "id" == header[0]  # type: ignore

        await awrite_eval_results_header(eval_str, has_id)

        async for row in csv_reader:
            task = asyncio.create_task(
                process_eval_row(semaphore, row, has_id=has_id, eval_str=eval_str, llm=llm, verbose=verbose)
            )
            tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    await run_in_threadpool(summarize_eval_results, eval_str)
