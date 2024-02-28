from typing import Any
from unittest.mock import patch

import pytest
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence

import heavyiq.lcel.chains.heavydb.sql_chain
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.lcel.chains import sql_chain as sql_chain_runnable


# Test case with a custom decorator
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data,config,llm_output",
    [
        (
            {
                "question": "How many states begin with the letter A? What are they?",
                "tables": ["usa_states"],
            },
            {"temperature": 0},
            "SELECT COUNT(DISTINCT STATE_NAME) AS count_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
        ),
        (
            {
                "question": "How many states begin with the letter A? What are they?",
                "tables": ["usa_states"],
            },
            {"temperature": 0},
            "MISSING_DATA: No data found containing the dates of Apple's earnings calls.",
        ),
    ],
)
async def test_should_pass_sql_chain(data: dict, config: dict, llm_output: str, session_id: str) -> None:
    request_data = data.copy()
    request_data["session_id"] = session_id

    with patch("heavyiq.lcel.chains.heavydb.sql_chain.aget_table_info_wrt_token_limit", return_value=""):
        async with heavydb_context(session_id):
            bounded: RunnableSequence = sql_chain_runnable.bound
            bounded.first = RunnablePassthrough().assign(
                sql_cmd=lambda x: llm_output,
                max_revisions=lambda x: 1,
            )
            # sql_chain = RunnableBinding(bounded)
            result = await sql_chain_runnable.ainvoke(request_data, config=config)
            if "SELECT" in result["query"]:
                assert result["query"] == llm_output
                assert not result["error"]
            elif "MISSING_DATA:" in result["query"] or "AMBIGUOUS_QUESTION" in result["query"]:
                assert result["error"] == llm_output
