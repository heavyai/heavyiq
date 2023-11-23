from uuid import uuid4
import os
import asyncio
import aiofiles
from aiocsv.readers import AsyncReader
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel

import click
from langsmith import Client
from heavyiq.config import get_config
from fastapi.concurrency import run_in_threadpool
from heavyiq.cli.decorators import coro
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_llm_by_type, LLMType
from heavyiq.logging_utils import get_heavyiq_logger

from .utils import (
    compute_prob_stats,
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
    enable_logprobs: bool = True,
    enable_querystats: bool = True,
    langsmith_client: Client | None = None,
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
        db = await run_in_threadpool(HeavyDB.from_env, db_name=db_id, include_tables=tables)
        chain = get_nl_to_sql_chain_by_llm(llm)(
            database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[eval_str, "cli"]
        )

        prob_stats = None
        query_stats = None
        try:
            res = await chain.acall({chain.input_key: question}, include_run_info=is_langsmith_active)
            pred_query = res[chain.output_key]
            logger.info(f"Generated SQL: {pred_query}")
            logger.debug("Evaluating SQL")

            if "logprobs" in res and len(res["logprobs"]) > 0:
                prob_stats = await run_in_threadpool(
                    compute_prob_stats, res["logprobs"]["tokens"], res["logprobs"]["top_logprobs"]
                )

            eval_res, query_stats = await asyncio.gather(
                run_in_threadpool(sql_rate_reply, gold_query, pred_query, db=db), db.aquery_stats(pred_query)
            )
            logger.info(f"Evaluation Success: {eval_res['success']}")
            logger.debug(f"Evaluation Status: {eval_res['status']}")
            if langsmith_client:
                feedback_id = str(res["__run"].run_id)
                logger.debug(f"Langsmith Run ID: {feedback_id}")
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
                question,
                gold_query,
                eval_res["success"],
                eval_res["status"],
                pred_query,
                query_id=query_id,
                error=eval_res["error"],
                prob_stats=prob_stats,
                query_stats=query_stats,
                enable_logprobs=enable_logprobs,
                enable_querystats=enable_querystats,
            )
        except NLtoSQLException as e:
            logger.exception(f"Failed to generate SQL: {e}")
            await awrite_eval_results_row(
                eval_str,
                db_id,
                question,
                gold_query,
                False,
                "failed_to_generate_sql",
                e.failed_sql,  # type: ignore
                query_id=query_id,
                prob_stats=prob_stats,
                query_stats=query_stats,
                enable_logprobs=enable_logprobs,
                enable_querystats=enable_querystats,
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

    if not await run_in_threadpool(os.path.exists, eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Eval ID: {eval_id}")

    tasks, semaphore = [], asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=temperature)

    config = get_config()
    langsmith_client = Client() if is_langsmith_active else None

    # csv must have columns: id (optional), db_id, tables, question, answer
    async with aiofiles.open(eval_dataset_csv, mode="r") as f:
        csv_reader = AsyncReader(f)
        header = await anext(csv_reader)
        has_id = "id" == header[0]  # type: ignore

        await awrite_eval_results_header(eval_str, has_id, config.enable_logprobs)

        async for row in csv_reader:
            task = asyncio.create_task(
                process_eval_row(
                    semaphore,
                    row,
                    has_id=has_id,
                    eval_str=eval_str,
                    llm=llm,
                    verbose=verbose,
                    enable_logprobs=config.enable_logprobs,
                    enable_querystats=True,
                    langsmith_client=langsmith_client,
                )
            )
            tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    await run_in_threadpool(summarize_eval_results, eval_str)
