import asyncio
import os
from uuid import uuid4

import aiofiles
import click
from aiocsv.readers import AsyncReader
from aiocsv.writers import AsyncWriter
from fastapi.concurrency import run_in_threadpool
from langchain.chat_models.base import BaseChatModel
from langchain.globals import set_verbose
from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import BaseModel
from langsmith import Client
from thrift.transport.TTransport import TTransportException

from heavyiq.cli.decorators import coro
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.logging_utils import get_heavyiq_logger

from .utils import (aextract_tables_from_query, awrite_eval_results_header,
                    awrite_eval_results_row,
                    check_predicted_query_equals_gold_query,
                    compute_prob_stats, sql_rate_reply, summarize_eval_results)


@click.group()
def eval():
    """Evaluate models."""
    pass


async def process_question(
    question,
    is_langsmith_active,
    chain,
    db,
    gold_query,
    logger,
    eval_str,
    db_id,
    query_id,
    enable_logprobs: bool = True,
    enable_querystats: bool = True,
    langsmith_client: Client | None = None,
):
    prob_stats, query_stats, pred_query = None, None, ""
    try:
        tables = await aextract_tables_from_query(db, gold_query)
        res = await chain.acall({chain.input_key: question, "tables": tables}, include_run_info=is_langsmith_active)
        pred_query = res[chain.output_key]
        logger.info(f"Generated SQL: {pred_query}")
        logger.debug("Evaluating SQL")

        if "logprobs" in res and len(res["logprobs"]) > 0:
            prob_stats = await run_in_threadpool(
                compute_prob_stats, res["logprobs"]["tokens"], res["logprobs"]["top_logprobs"]
            )

        eval_res = await sql_rate_reply(gold_query, pred_query, db=db, question=question)
        if eval_res.get("error"):
            del db
            db = await HeavyDB.from_env_async(db_id)
        query_stats = await db.aquery_stats(pred_query)
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
        await awrite_eval_results_row(
            eval_str,
            db_id,
            question,
            gold_query,
            False,
            str(e),
            pred_query,  # type: ignore
            query_id=query_id,
            prob_stats=prob_stats,
            query_stats=query_stats,
            enable_logprobs=enable_logprobs,
            enable_querystats=enable_querystats,
        )


async def get_db(dbname: str) -> HeavyDB:
    try:
        db = await HeavyDB.from_env_async(dbname)
    except Exception:
        await asyncio.sleep(1)
        db = await HeavyDB.from_env_async(dbname)
    return db


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
        query_id, db_id, question, gold_query = row
    else:
        query_id = None
        db_id, question, gold_query = row
    async with semaphore:
        logger.info(f"Processing Question: {question}")
        db = await get_db(db_id)
        async with heavydb_context(db):
            chain = get_nl_to_sql_chain_by_llm(llm)(
                database=db, llm=llm, callbacks=None if verbose else [], verbose=verbose, tags=[eval_str, "cli"]
            )

            await process_question(
                question,
                is_langsmith_active,
                chain,
                db,
                gold_query,
                logger,
                eval_str,
                db_id,
                query_id,
                enable_logprobs=enable_logprobs,
                enable_querystats=enable_querystats,
                langsmith_client=langsmith_client,
            )


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

    tasks, semaphore = [], asyncio.Semaphore(30)  # Limit to 10 concurrent tasks

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

    await asyncio.gather(*tasks, return_exceptions=False)

    await run_in_threadpool(summarize_eval_results, eval_str)


@eval.command()
@coro
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.option("--batch-size", default=10, help="How many questions to process parallelly?", type=int)
@click.argument("eval_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_tables(
    ctx: click.Context, eval_dataset_csv: str, batch_size: int, temperature: float, verbose: bool
) -> None:
    from heavyiq.langchain.utils import is_langsmith_active
    from heavyiq.lcel.chains.eval.eval_table_chain import chain

    logger = get_heavyiq_logger()

    if not await run_in_threadpool(os.path.exists, eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Dataset ID: {eval_id}")

    rows_written: int = 0
    eval_results_csv: str = f"./eval/results/{eval_str}_results.csv"

    # csv must have columns: id (optional), db_id, tables, question, answer
    async with aiofiles.open(eval_dataset_csv, mode="r") as f, aiofiles.open(eval_results_csv, "a", newline="") as wf:
        csv_reader, writer = AsyncReader(f), AsyncWriter(wf, dialect="unix")
        header = await anext(csv_reader)
        has_id = "id" == header[0]  # type: ignore

        # write header
        header_data = ["db_id", "gold_tables", "pred_tables", "success", "status", "error"]
        if has_id:
            header_data = ["id"] + header_data
        await writer.writerow(header_data)

        while True:
            rows = []
            for _ in range(batch_size):
                try:
                    rows.append(await csv_reader.__anext__())
                except StopAsyncIteration:
                    pass
            if not rows:
                break
            print(f"Processing {len(rows)} rows...")
            batch_inputs = []
            for row in rows:
                if has_id:
                    query_id, db_id, question, answer = row
                else:
                    query_id = None
                    db_id, question, answer = row
                batch_inputs.append({"query_id": query_id, "db_id": db_id, "question": question, "sql": answer})
            if not batch_inputs:
                break

            chain_output = await chain.abatch(batch_inputs, config={"configurable": {"llm_temperature": temperature}})
            row_datas = []
            for data in chain_output:
                row_datas.append(list(data.values()))
            if row_datas:
                await writer.writerows(row_datas)
                rows_written += len(row_datas)

    await run_in_threadpool(summarize_eval_results, eval_str)


@eval.command()
@coro
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.option("--enable-query-stats", default=True, help="Enable Query Stats?", type=bool)
@click.option("--batch-size", default=10, help="How many questions to process parallelly?", type=int)
@click.argument("eval_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_auto_questions(
    ctx: click.Context,
    eval_dataset_csv: str,
    batch_size: int,
    temperature: float,
    verbose: bool,
    enable_query_stats: bool,
) -> None:
    from heavyiq.langchain.utils import is_langsmith_active
    from heavyiq.lcel.chains.eval.eval_auto_table_question_chain import chain

    logger = get_heavyiq_logger()

    if not await run_in_threadpool(os.path.exists, eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Dataset ID: {eval_id}")
    has_id: bool = True
    processor_count = batch_size

    async def producer(queue: asyncio.Queue, input_file_path: str):
        """
        producer responsible for extracting the records from csv and put it in the queue.
        """
        nonlocal has_id
        async with aiofiles.open(input_file_path, mode="r") as f:
            csv_reader = AsyncReader(f)
            header = await anext(csv_reader)
            has_id = "id" == header[0]
            async for row in csv_reader:
                await queue.put(row)
                await asyncio.sleep(0.1)

        print("Successfully filled the input queue.")
        # await queue.put(None)  # Sentinel value to signal the end of input
        # Signal consumers to stop
        for _ in range(processor_count):
            await queue.put(None)

    async def processor(input_queue: asyncio.Queue, output_queue: asyncio.Queue, processor_id: int):
        """
        Trigger the undelying chain to produce answer.
        """
        while True:
            prob_stats, query_stats = None, None
            item = await input_queue.get()

            if item is None:
                print(f"Processor {processor_id} stopped!")
                await output_queue.put(None)
                await asyncio.sleep(0.1)
                input_queue.task_done()
                break

            if has_id:
                query_id, db_id, question, gold_query, *optional_gold_queries = item
            else:
                query_id = None
                db_id, question, gold_query, *optional_gold_queries = item

            logger.info(f'Processing "{question}"...')

            chain_inputs = {
                "query_id": query_id,
                "db_id": db_id,
                "question": question,
                "sql": gold_query,
                "enable_query_stats": enable_query_stats,
                "gold_queries": optional_gold_queries,
            }

            db = await HeavyDB.from_env_async(db_id)
            async with heavydb_context(db):
                chain_output = await chain.ainvoke(
                    chain_inputs, config={"configurable": {"llm_temperature": temperature}}
                )
            query_stats = chain_output["query_stats"].values()
            if not query_stats:
                query_stats = [None, None, None, None, None]

            output_item = (
                query_id,
                db_id,
                question,
                gold_query,
                chain_output["pred_query"],
                chain_output["success"],
                chain_output["status"],
                chain_output["error"],
                prob_stats,
                query_stats,
            )
            await output_queue.put(output_item)
            await asyncio.sleep(0.1)
            input_queue.task_done()

    async def build_row_data(item: tuple) -> list:
        """
        Builds output csv record from the item we got from the output queue.
        """
        (
            query_id,
            db_id,
            question,
            gold_query,
            pred_query,
            eval_res_success,
            eval_res_status,
            eval_res_error,
            prob_stats,
            query_stats,
        ) = item
        row_data = []
        if query_id:
            row_data.append(query_id)

        row_data.extend(
            [db_id, question, gold_query, pred_query, eval_res_success, eval_res_status, eval_res_error or ""]
        )
        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]
        return row_data

    async def consumer(output_queue: asyncio.Queue, output_file_path: str, enable_query_stats: bool = False):
        """
        consumer
        """
        await asyncio.sleep(1)  # initial sleep to identify wherher the eval dataset contain header or not
        async with aiofiles.open(output_file_path, "a", newline="") as wf:
            writer = AsyncWriter(wf, dialect="unix")

            # write header
            header_data = ["db_id", "question", "gold_query", "pred_query", "success", "status", "error"]
            if has_id:
                header_data = ["id"] + header_data
            if enable_query_stats:
                header_data.extend(["num_joins", "num_unions", "num_aggs", "num_filters", "num_sorts"])
            await writer.writerow(header_data)
            len_nones = 0
            while True:
                item = await output_queue.get()
                if item is None:
                    len_nones += 1
                    if len_nones >= processor_count:
                        output_queue.task_done()
                        print("consumer stopped")
                        break
                    output_queue.task_done()
                    continue
                row = await build_row_data(item)
                await writer.writerow(row)
                output_queue.task_done()

    eval_results_csv: str = f"./eval/results/{eval_str}_results.csv"
    input_queue: asyncio.Queue = asyncio.Queue()  # contain csv record
    output_queue: asyncio.Queue = asyncio.Queue()  # contain data to be written on output csv

    producer_task = asyncio.create_task(producer(input_queue, eval_dataset_csv))
    processors = [
        asyncio.create_task(processor(input_queue, output_queue, processor_id))
        for processor_id in range(processor_count)
    ]
    consumer_task = asyncio.create_task(consumer(output_queue, eval_results_csv, enable_query_stats=False))

    await input_queue.join()
    await output_queue.join()

    await asyncio.gather(producer_task, *processors, consumer_task)
    await run_in_threadpool(summarize_eval_results, eval_str)


@eval.command()
@coro
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--n", default=1, help="number of generations", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--generate", "-g", is_flag=True, help="Pick an sql query from the list of generated sql queries")
@click.argument("eval_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_questions_lcel(
    ctx: click.Context,
    eval_dataset_csv: str,
    generate: bool,
    verbose: bool,
    n: int,
    temperature: float,
):
    """
    Run config model on questions using lcel approach.
    """
    from heavyiq.lcel.chains import sql_chain, sql_gen_chain

    if generate:
        chain = sql_gen_chain
    else:
        chain = sql_chain

    set_verbose(verbose)
    logger = get_heavyiq_logger()
    if not await run_in_threadpool(os.path.exists, eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Dataset ID: {eval_id}")
    has_id: bool = True
    processor_count = 30

    async def producer(queue: asyncio.Queue, input_file_path: str):
        """
        producer responsible for extracting the records from csv and put it in the queue.
        """
        nonlocal has_id
        async with aiofiles.open(input_file_path, mode="r") as f:
            csv_reader = AsyncReader(f)
            header = await anext(csv_reader)
            has_id = "id" == header[0]
            async for row in csv_reader:
                await queue.put(row)
                await asyncio.sleep(0.1)

        print("Successfully filled the input queue.")
        # await queue.put(None)  # Sentinel value to signal the end of input
        # Signal consumers to stop
        for _ in range(processor_count):
            await queue.put(None)

    async def predict_query(question: str, gold_query: str, db: HeavyDB) -> tuple[str, str]:
        """
        Predicts the target query.

        Args:
            db: HeavyDB instance
            gold_query: golden query

        Returns:
            predicted_query and error
        """
        out: dict = {"query": None, "error": None}
        async with heavydb_context(db):
            tables = await aextract_tables_from_query(db, gold_query)
            out = await chain.ainvoke(
                {"question": question, "tables": tables, "session_id": db._conn._session},
                config={"configurable": {"llm_n": n, "llm_temperature": temperature}},
            )
        return out["query"], out["error"]

    async def processor(input_queue: asyncio.Queue, output_queue: asyncio.Queue, processor_id: int):
        """
        processing each record takes place here.
        """
        while True:
            prob_stats, query_stats = None, None
            item = await input_queue.get()

            if item is None:
                print(f"Processor {processor_id} stopped!")
                await output_queue.put(None)
                await asyncio.sleep(0.1)
                input_queue.task_done()
                break

            optional_gold_queries = []
            if has_id:
                query_id, db_id, question, gold_query, *optional_gold_queries = item
            else:
                query_id = None
                db_id, question, gold_query, *optional_gold_queries = item

            db = await HeavyDB.from_env_async(db_id)
            pred_query, query_error = await predict_query(question, gold_query, db)

            if not query_error:
                try:
                    gold_queries = [gold_query] + optional_gold_queries
                    for gold in gold_queries:
                        if not gold:
                            continue
                        if check_predicted_query_equals_gold_query(pred_query, gold):
                            gold_query = gold
                            break
                    eval_res = await sql_rate_reply(gold_query, pred_query, db=db, question=question)
                    if eval_res.get("error"):
                        del db
                        db = await HeavyDB.from_env_async(db_id)
                    query_stats = await db.aquery_stats(pred_query)

                except Exception as e:
                    eval_res_success = False
                    eval_res_status = "failed_to_generate_sql"
                    error = f"Failed to calculate query stats, {e}"
                    eval_res_error = error
                    logger.error(error)
                else:
                    eval_res_success = eval_res["success"]
                    eval_res_status = eval_res["status"]
                    eval_res_error = eval_res["error"]
                    logger.info(f"Evaluation Success: {eval_res['success']}")
            else:
                eval_res_success = False
                eval_res_status = "failed_to_generate_sql"
                eval_res_error = query_error
                logger.error("Error occurs while predicting the query.")

            output_item = (
                query_id,
                db_id,
                question,
                gold_query,
                pred_query,
                eval_res_success,
                eval_res_status,
                eval_res_error,
                prob_stats,
                query_stats,
            )

            await output_queue.put(output_item)
            await asyncio.sleep(0.1)
            input_queue.task_done()

    async def build_row_data(item: tuple) -> list:
        """
        Builds output csv record from the item we got from the output queue.
        """
        (
            query_id,
            db_id,
            question,
            gold_query,
            pred_query,
            eval_res_success,
            eval_res_status,
            eval_res_error,
            prob_stats,
            query_stats,
        ) = item
        row_data = []
        if query_id:
            row_data.append(query_id)

        row_data.extend(
            [db_id, question, gold_query, pred_query, eval_res_success, eval_res_status, eval_res_error or ""]
        )
        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]
        return row_data

    async def consumer(output_queue: asyncio.Queue, output_file_path: str, enable_query_stats: bool = False):
        """
        consumer
        """
        await asyncio.sleep(1)  # initial sleep to identify wherher the eval dataset contain header or not
        async with aiofiles.open(output_file_path, "a", newline="") as wf:
            writer = AsyncWriter(wf, dialect="unix")

            # write header
            header_data = ["db_id", "question", "gold_query", "pred_query", "success", "status", "error"]
            if has_id:
                header_data = ["id"] + header_data
            if enable_query_stats:
                header_data.extend(["num_joins", "num_unions", "num_aggs", "num_filters", "num_sorts"])
            await writer.writerow(header_data)
            len_nones = 0
            while True:
                item = await output_queue.get()
                if item is None:
                    len_nones += 1
                    if len_nones >= processor_count:
                        output_queue.task_done()
                        print("consumer stopped")
                        break
                    output_queue.task_done()
                    continue
                row = await build_row_data(item)
                await writer.writerow(row)
                output_queue.task_done()

    eval_results_csv: str = f"./eval/results/{eval_str}_results.csv"
    input_queue: asyncio.Queue = asyncio.Queue()  # contain csv record
    output_queue: asyncio.Queue = asyncio.Queue()  # contain data to be written on output csv

    producer_task = asyncio.create_task(producer(input_queue, eval_dataset_csv))
    processors = [
        asyncio.create_task(processor(input_queue, output_queue, processor_id))
        for processor_id in range(processor_count)
    ]
    consumer_task = asyncio.create_task(consumer(output_queue, eval_results_csv, enable_query_stats=False))

    await input_queue.join()
    await output_queue.join()

    await asyncio.gather(producer_task, *processors, consumer_task)
    await run_in_threadpool(summarize_eval_results, eval_str)


@eval.command()
@coro
@click.option("--temperature", default=0.0, help="Temperature for LLM (Defaults to 0.0)", type=float)
@click.option("--verbose", default=False, help="Verbose output", type=bool)
@click.argument("eval_dataset_csv", type=str)
@click.pass_context  # type: ignore
async def run_config_model_on_questions_cot_lcel(
    ctx: click.Context, eval_dataset_csv: str, temperature: float, verbose: bool
):
    """
    Run config model on questions using lcel approach.
    """
    from heavyiq.lcel.chains import sql_cot_chain

    set_verbose(verbose)
    logger = get_heavyiq_logger()
    if not await run_in_threadpool(os.path.exists, eval_dataset_csv):
        raise Exception(f"eval_dataset_csv does not exist: {eval_dataset_csv}")

    eval_id = uuid4().hex[:8]
    eval_str = f"eval_{eval_id}"
    logger.info(f"Dataset ID: {eval_id}")
    has_id: bool = True
    processor_count = 30

    async def producer(queue: asyncio.Queue, input_file_path: str):
        """
        producer responsible for extracting the records from csv and put it in the queue.
        """
        nonlocal has_id
        async with aiofiles.open(input_file_path, mode="r") as f:
            csv_reader = AsyncReader(f)
            header = await anext(csv_reader)
            has_id = "id" == header[0]
            async for row in csv_reader:
                await queue.put(row)
                await asyncio.sleep(0.1)

        print("Successfully filled the input queue.")
        # await queue.put(None)  # Sentinel value to signal the end of input
        # Signal consumers to stop
        for _ in range(processor_count):
            await queue.put(None)

    async def predict_query(question: str, gold_query: str, db: HeavyDB) -> tuple[str, str]:
        """
        Predicts the target query.

        Args:
            db: HeavyDB instance
            gold_query: golden query

        Returns:
            predicted_query and error
        """
        out: dict = {"query": None, "error": None}
        async with heavydb_context(db):
            tables = await aextract_tables_from_query(db, gold_query)
            out = await sql_cot_chain.ainvoke({"question": question, "tables": tables, "session_id": db._conn._session})
        return out["query"], out["error"], out["cot"]

    async def processor(input_queue: asyncio.Queue, output_queue: asyncio.Queue, processor_id: int):
        """
        processing each record takes place here.
        """
        while True:
            prob_stats, query_stats = None, None
            item = await input_queue.get()

            if item is None:
                print(f"Processor {processor_id} stopped!")
                await output_queue.put(None)
                await asyncio.sleep(0.1)
                input_queue.task_done()
                break

            optional_gold_queries = []
            if has_id:
                query_id, db_id, question, gold_query, *optional_gold_queries = item
            else:
                query_id = None
                db_id, question, gold_query, *optional_gold_queries = item

            db = await HeavyDB.from_env_async(db_id)
            pred_query, query_error, cot = await predict_query(question, gold_query, db)

            if not query_error:
                try:
                    gold_queries = [gold_query] + optional_gold_queries
                    for gold in gold_queries:
                        if not gold:
                            continue
                        if check_predicted_query_equals_gold_query(pred_query, gold):
                            gold_query = gold
                            break
                    eval_res = await sql_rate_reply(gold_query, pred_query, db=db, question=question)
                    if eval_res.get("error"):
                        del db
                        db = await HeavyDB.from_env_async(db_id)
                    query_stats = await db.aquery_stats(pred_query)

                except Exception as e:
                    eval_res_success = False
                    eval_res_status = "failed_to_generate_sql"
                    error = f"Failed to calculate query stats, {e}"
                    eval_res_error = error
                    logger.error(error)
                else:
                    eval_res_success = eval_res["success"]
                    eval_res_status = eval_res["status"]
                    eval_res_error = eval_res["error"]
                    logger.info(f"Evaluation Success: {eval_res['success']}")
            else:
                eval_res_success = False
                eval_res_status = "failed_to_generate_sql"
                eval_res_error = query_error
                logger.error("Error occurs while predicting the query.")

            output_item = (
                query_id,
                db_id,
                question,
                gold_query,
                pred_query,
                eval_res_success,
                eval_res_status,
                eval_res_error,
                prob_stats,
                query_stats,
                cot,
            )

            await output_queue.put(output_item)
            await asyncio.sleep(0.1)
            input_queue.task_done()

    async def build_row_data(item: tuple) -> list:
        """
        Builds output csv record from the item we got from the output queue.
        """
        (
            query_id,
            db_id,
            question,
            gold_query,
            pred_query,
            eval_res_success,
            eval_res_status,
            eval_res_error,
            prob_stats,
            query_stats,
            cot,
        ) = item
        row_data = []
        if query_id:
            row_data.append(query_id)

        row_data.extend(
            [db_id, question, gold_query, pred_query, eval_res_success, eval_res_status, eval_res_error or "", cot]
        )
        row_data = [str(item).replace("\n", " ").replace("\r", " ") for item in row_data]
        return row_data

    async def consumer(output_queue: asyncio.Queue, output_file_path: str, enable_query_stats: bool = False):
        """
        consumer
        """
        await asyncio.sleep(1)  # initial sleep to identify wherher the eval dataset contain header or not
        async with aiofiles.open(output_file_path, "a", newline="") as wf:
            writer = AsyncWriter(wf, dialect="unix")

            # write header
            header_data = ["db_id", "question", "gold_query", "pred_query", "success", "status", "error", "cot"]
            if has_id:
                header_data = ["id"] + header_data
            if enable_query_stats:
                header_data.extend(["num_joins", "num_unions", "num_aggs", "num_filters", "num_sorts"])
            await writer.writerow(header_data)
            len_nones = 0
            while True:
                item = await output_queue.get()
                if item is None:
                    len_nones += 1
                    if len_nones >= processor_count:
                        output_queue.task_done()
                        print("consumer stopped")
                        break
                    output_queue.task_done()
                    continue
                row = await build_row_data(item)
                await writer.writerow(row)
                output_queue.task_done()

    eval_results_csv: str = f"./eval/results/{eval_str}_results.csv"
    input_queue: asyncio.Queue = asyncio.Queue()  # contain csv record
    output_queue: asyncio.Queue = asyncio.Queue()  # contain data to be written on output csv

    producer_task = asyncio.create_task(producer(input_queue, eval_dataset_csv))
    processors = [
        asyncio.create_task(processor(input_queue, output_queue, processor_id))
        for processor_id in range(processor_count)
    ]
    consumer_task = asyncio.create_task(consumer(output_queue, eval_results_csv, enable_query_stats=False))

    await input_queue.join()
    await output_queue.join()

    await asyncio.gather(producer_task, *processors, consumer_task)
    await run_in_threadpool(summarize_eval_results, eval_str)
