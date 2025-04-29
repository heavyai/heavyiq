"""
Script for generating training data for the model to generate Vega Lite Spec.
$ python generate_vega_training_data.py -i data.csv -o output.csv -n 3
"""

import argparse
import asyncio
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, TypedDict

import aiocsv
import aiofiles
import pandas as pd
import uvloop

# CONFIG
CSV_FILE = "data.csv"
OUTPUT_CSV_FILE = "output.csv"
CHUNK_SIZE = 1  # chunk size where each process is supposed to be handled
NUM_PROCESSES = 5  # number of processes to be spawned
QUEUE_MAXSIZE = 10  # limit memory usage
MODEL_NAME = "gemini-2.0-flash"

# Use uvloop for improved performance on UNIX-based systems
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class WriteRow(TypedDict):
    table_name: str
    database_name: str
    question: str
    query: str
    is_valid: bool
    vega_spec: str
    image_path: str


# ========= Async per-record processing (CPU-bound simulation) =========
async def process_record(record: Dict[str, Any], model_name: str) -> list[WriteRow]:
    from heavyiq.training.vega_lite.chart_graph import app as vega_graph

    pid = os.getpid()
    tid = threading.get_ident()
    print(f"[Async Function] (PID={pid}, TID={tid}) Processing record ID {record.get('id')}")
    # create images dir if not exists
    os.makedirs("images", exist_ok=True)
    # Simulate asynchronous work (for example, an I/O operation or async CPU work)
    count = int(record.get("count", 1))
    inputs = {
        "database_name": record["database_name"],
        "table_name": record["table_name"],
        "n": count,
        "model_name": model_name,
        "image_folder": "images",
        "input_viz_question": record.get("question", ""),
        "input_sql": record.get("sql", ""),
    }

    final_state = await vega_graph.ainvoke(inputs)
    output_rows = []
    for line in final_state["questions_with_vega_image"]:
        question, query, vega_spec, image_path = line
        if isinstance(vega_spec, dict):
            vega_spec = json.dumps(vega_spec)

        output_rows.append(
            WriteRow(
                database_name=record["database_name"],
                table_name=record["table_name"],
                question=question,
                query=query,
                is_valid=True,
                vega_spec=vega_spec,
                image_path=image_path,
            )
        )
    for line in final_state["questions_with_error_sql"]:
        question, query = line
        output_rows.append(
            WriteRow(
                database_name=record["database_name"],
                table_name=record["table_name"],
                question=question,
                query=query,
                is_valid=False,
                vega_spec="",
                image_path="",
            )
        )

    return output_rows


# ========= Per-process chunk processor =========
def worker_process(chunk: List[Dict[str, Any]], model_name: str) -> List[WriteRow]:
    pid = os.getpid()
    print(f"[Process {pid}] Received chunk with {len(chunk)} records")
    # Create a new event loop for this process
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    async def handle_chunk():
        # Create tasks for each record using the async process_record
        tasks = [process_record(record, model_name) for record in chunk]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = loop.run_until_complete(handle_chunk())
    loop.close()

    # collect all the errors and re-raise it
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(f"{len(errors)} errors in worker process: {errors}")
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results


# ========= Producer coroutine =========
async def producer(queue: asyncio.Queue):
    print("[Producer] Starting to read CSV and enqueue chunks...")
    async with aiofiles.open(CSV_FILE, mode="r", newline="") as afp:
        reader = aiocsv.AsyncDictReader(afp)
        chunk = []
        async for row in reader:
            chunk.append(row)
            if len(chunk) == CHUNK_SIZE:
                print(f"[Producer] Enqueuing chunk of size {CHUNK_SIZE}")
                await queue.put(chunk)
                chunk = []
        if chunk:
            print(f"[Producer] Enqueuing final chunk of size {len(chunk)}")
            await queue.put(chunk)
    # Signal completion (push 'None' poison pills for each consumer)
    for _ in range(NUM_PROCESSES):
        await queue.put(None)
    print("[Producer] Finished producing chunks and sent stop signals.")


# ========= Consumer coroutine =========
async def consumer(queue: asyncio.Queue, result_list: List, process_pool: ProcessPoolExecutor, cid: int):
    print(f"[Consumer-{cid}] Started")
    loop = asyncio.get_event_loop()
    while True:
        chunk = await queue.get()
        if chunk is None:
            print(f"[Consumer-{cid}] Received stop signal.")
            queue.task_done()
            break
        print(f"[Consumer-{cid}] Dequeued chunk of size {len(chunk)}")
        try:
            # Process the chunk in a separate process
            result = await loop.run_in_executor(process_pool, worker_process, chunk, MODEL_NAME)
            result_list.extend(result)
        except Exception as e:
            print(f"[Consumer-{cid}] ❌ Worker failed with exception: {e}")
        # don't break the loop on exception since we have one item to process ie. None
        queue.task_done()
    print(f"[Consumer-{cid}] Queue size: {queue.qsize()}")
    print(f"[Consumer-{cid}] Exiting.")


# ========= Async Main =========
async def async_main():
    from heavyiq.langchain.utils import init_telemetrics

    print("[Main] Starting async processing pipeline...")
    # generate_sample_csv(CSV_FILE, rows=100)
    # Init telemetrics
    init_telemetrics()
    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    results = []
    process_pool = ProcessPoolExecutor(max_workers=NUM_PROCESSES)

    # Launch producer coroutine
    producer_task = asyncio.create_task(producer(queue))

    # Launch consumer coroutines (with consumer id for logging)
    consumer_tasks = [asyncio.create_task(consumer(queue, results, process_pool, cid)) for cid in range(NUM_PROCESSES)]

    await asyncio.gather(producer_task)
    await queue.join()  # Wait until all items in the queue are processed

    for c in consumer_tasks:
        await c  # Ensure each consumer has exited

    if not results:
        return None

    df = pd.DataFrame(results)
    df = df.sort_values(by=["database_name", "table_name"])
    df["id"] = range(1, len(df) + 1)
    # Define desired column order
    column_order = ["id", "database_name", "table_name", "question", "query", "is_valid", "image_path", "vega_spec"]
    df.to_csv(OUTPUT_CSV_FILE, columns=column_order, index=False)
    print("\n[Main] Final Merged Output (First 5 Rows):")
    print(df.head())
    print(f"Successfully written output to {OUTPUT_CSV_FILE}")


def main():
    global CSV_FILE, OUTPUT_CSV_FILE, MODEL_NAME
    parser = argparse.ArgumentParser(description="Generate training data for Vega-Lite spec model.")
    parser.add_argument("-i", "--input", help="Input csv file path", default=CSV_FILE)
    parser.add_argument("-o", "--output", help="Output csv file path", default=OUTPUT_CSV_FILE)
    parser.add_argument("-m", "--model", help="LLM Model Name", default=MODEL_NAME)

    args = parser.parse_args()

    CSV_FILE, OUTPUT_CSV_FILE, MODEL_NAME = args.input, args.output, args.model

    asyncio.run(async_main())


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total seconds: {end-start}")
