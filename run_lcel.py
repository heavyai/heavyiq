# Script which helps to run LCEL runnable chains
# You can play with any defined functions by passing heavydb_session id to each
# python run_lcel.py <heavydb_session_id>

import asyncio
import sys

from heavyiq.lcel.chains.heavydb.answer_chain import chain as nl_to_answer_chain
from heavyiq.lcel.chains.heavydb.sql_chain import chain as nl_to_sql_chain


async def nl_to_sql(session_id: str):
    out = await nl_to_sql_chain.ainvoke(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        }
    )
    print(out)


async def nl_to_sql_stream(session_id: str):
    async for s in nl_to_sql_chain.astream(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        }
    ):
        print(s, end="", flush=True)


async def nl_to_sql_stream_intermediate_steps(session_id: str):
    async for s in nl_to_sql_chain.astream_log(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        }
    ):
        print("-" * 40)
        print(s)


async def nl_to_sql_stream_intermediate_steps_in_incremental_fashion(session_id: str):
    """
    You can simply pass diff=False to get incremental values of RunState. You get more verbose output with more repetitive parts.
    """
    async for s in nl_to_sql_chain.astream_log(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        },
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
    out = await nl_to_answer_chain.ainvoke(
        {
            "question": "How many states begin with the letter A? What are they?",
            "tables": ["usa_states"],
            "session_id": session_id,
        }
    )
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
    out = await nl_to_answer_chain.abatch(inputs)
    print(out)


if __name__ == "__main__":
    session_id = sys.argv[1]  # type: ignore
    asyncio.run(nl_to_answer_batch(session_id))
