# Script which helps to run LCEL runnable chains
# You can play with any defined functions by passing heavydb_session id to each
# python run_lcel.py <heavydb_session_id>

import asyncio
import sys
import time
from contextvars import ContextVar

from langchain import callbacks

from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.langchain.logging import log_chain_runnable
from heavyiq.langchain.utils import init_telemetrics
from heavyiq.lcel.chains.heavydb.answer_chain import chain as nl_to_answer_chain
from heavyiq.lcel.chains.heavydb.sql_chain import chain as nl_to_sql_chain
from heavyiq.lcel.chains.heavydb.sql_chain import query_runnable

# from heavyiq.lcel.chains.heavydb.sql_chain import complete_chain as nl_to_sql_complete_chain

# init_telemetrics()

input_ctx_var: ContextVar[dict] = ContextVar("input_ctx_var", default={})


async def nl_to_sql(session_id: str):
    async with heavydb_context(session_id):  # type: ignore
        out = await nl_to_sql_chain.ainvoke(input_ctx_var.get())
        print(out)


async def nl_to_sql_with_openai_cb(session_id: str):
    out = await log_chain_runnable(
        nl_to_sql_chain,
        input_ctx_var.get(),
    )
    print(out)


async def nl_to_sql_with_cb_collect_runs(session_id: str):
    with callbacks.collect_runs() as cb:
        out = await nl_to_sql_chain.ainvoke(input_ctx_var.get())
        assert len(cb.traced_runs) == 1


async def nl_to_sql_stream(session_id: str):
    async for s in nl_to_sql_chain.astream(input_ctx_var.get()):
        print(s, end="", flush=True)


async def nl_to_sql_stream_intermediate_steps(session_id: str):
    async for s in nl_to_sql_chain.astream_log(input_ctx_var.get()):
        print("-" * 40)
        print(s)


async def nl_to_sql_logprobs(session_id: str):
    final_output, logprobs = "", {}
    async for log in nl_to_sql_chain.astream_log(input_ctx_var.get(), include_types=["llm"]):
        for op in log.ops:
            if op["path"] == "/final_output":
                final_output = op["value"]["output"]
            if "/final_output" in op["path"]:
                op_value = op["value"]
                if "generations" in op_value:
                    try:
                        logprobs = op_value["generations"][0][0]["generation_info"]["logprobs"]
                    except (KeyError, IndexError):
                        pass

    print(logprobs)
    print(final_output)


async def nl_to_sql_stream_intermediate_steps_in_incremental_fashion(session_id: str):
    """
    You can simply pass diff=False to get incremental values of RunState. You get more verbose output with more repetitive parts.
    """
    async for s in nl_to_sql_chain.astream_log(
        input_ctx_var.get(),
        diff=False,
    ):
        print("-" * 40)
        print(s)


async def nl_to_sql_batch(session_id: str):
    """
    Runnable can operate on batches of input.
    """
    questions = [
        "What is the total population in the USA according to the data in the usa_states table?",
        "Which state has the highest population density?",
        "Which state has the highest number of vacant housing units?",
    ]
    inputs = [
        {
            "question": q,
            "tables": ["usa_states"],
            "session_id": session_id,
        }
        for q in questions
    ]
    out = await nl_to_sql_chain.abatch(inputs)
    print(out)


async def nl_to_answer(session_id: str):
    async with heavydb_context(session_id):
        out = await nl_to_answer_chain.ainvoke(input_ctx_var.get())
        print(out)


async def nl_to_answer_batch(session_id: str):
    """
    Runnable can operate on batches of input.
    """
    questions = [
        "What is the total population in the USA according to the data in the usa_states table?",
        "Which state has the highest population density?",
        "Which state has the highest number of vacant housing units?",
    ]
    inputs = [
        {
            "question": q,
            "tables": ["usa_states"],
            "session_id": session_id,
        }
        for q in questions
    ]
    async with heavydb_context(session_id):
        out = await nl_to_answer_chain.abatch(inputs)
        print(out)


if __name__ == "__main__":
    start_time = time.perf_counter()
    session_id = sys.argv[1]  # type: ignore
    input_ctx_var.set(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        }
    )
    asyncio.run(nl_to_answer(session_id))
    end_time = time.perf_counter()
    print("Elapsed time during the whole program in seconds:", end_time - start_time)
