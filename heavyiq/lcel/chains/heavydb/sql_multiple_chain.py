# Generate multiple SQL queries
# Pass them to judge llm which is supposed to produce 1 and 0 for valid sql queries
# Pick the right one having the highest probability
import math
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.lcel.prompts import multiple_sql_judge_prompt

# from .sql_chain import table_info_runnable_lambda
from heavyiq.lcel.runnables.base import LLMGeneratorRunnable, SessionRunnable
from heavyiq.lcel.types import SqlChainInputType, SqlMultipleChainOutputType

from .sql_gen_chain import filter_valid_queries_chain as sql_generator_chain
from .sql_gen_chain import gen_query_variables as query_variables

CONFIG = get_config()


judge_llm_rbl = get_llm_by_type(LLMType.NL_TO_MULTIPLE_SQL_JUDGE, temperature=0.0)

judge_llm: Runnable = LLMGeneratorRunnable(llm_runnable=judge_llm_rbl, name="sql-judge-llm").with_config(
    config={
        "tags": ["intermediate-step"],
        "run_name": "Judge LLM",
        "metadata": {"step": "Invoking the LLM for judging mutiple SQL queries."},
    }
)
judge_prompt = multiple_sql_judge_prompt.with_config(config={"tags": ["judge_prompt"]})


async def filter_valid_queries(inputs: dict) -> list[str]:
    """
    Retrun only the valid SQL queries.
    """
    heavydb = await get_db(inputs["session_id"])
    valid_queries = []
    for query in set(inputs["sqls"]):
        try:
            await heavydb.avalidate_query(query)
        except Exception:
            pass
        else:
            valid_queries.append(query)
    return valid_queries


def prepare_judge_llm_context(inputs: dict) -> str:
    """
    Prepare the context str for judge llm which looks similar to below.

    SQL 1:
    <SQL query 1>

    SQL 2:
    <SQL query 2>

    …

    SQL 5:
    <SQL query 5>
    """
    context_str = ""
    for i, sql in enumerate(inputs.get("valid_sqls"), start=1):
        context_str += f"SQL {i}:\n{sql}\n"
    return context_str.strip()


def find_probability_score_from_logprobs(logprobs: dict) -> list[float]:
    """
    Helps to find probability score for a specific output format using logprobs data.
    """
    previous_token, predicted_tokens_probability = (), []
    for token, token_logprobs in zip(logprobs["tokens"], logprobs["token_logprobs"]):
        if token == "\n" or token == "":
            if previous_token and (previous_token[0] in ["1", "0"]):
                if previous_token[0] == "1":
                    probability = math.exp(previous_token[1])
                else:
                    probability = 1 - math.exp(previous_token[1])

                predicted_tokens_probability.append(probability)
        previous_token = (token, token_logprobs)
    return predicted_tokens_probability


def parse_judge_llm_output_and_assign_scores(inputs: dict) -> dict[str, list[tuple[str, float]]]:
    """
    Helps to assign SQL scores based on log probability.
    """
    valid_sqls = inputs["valid_sqls"]
    logprobs = inputs["judge_llm_output"][0]["logprobs"]
    scores = find_probability_score_from_logprobs(logprobs)
    # return list of queries in descending order acc to score
    sql_score_tuple_list = list(zip(valid_sqls, scores))[: len(valid_sqls)]
    return {"queries": sorted(sql_score_tuple_list, key=lambda x: x[1], reverse=True)}


def apply_top_k_and_cutoff(inputs: dict) -> dict[str, list[tuple[str, float]]]:
    """
    Apply cutoff threshold and top-k.
    """
    sqls = inputs["queries"]
    config, cutoff, top_k = get_config(), None, None
    if not cutoff:
        cutoff = config.custom_llm_api_nl_to_multiple_sql_judge_cutoff_score
    if not top_k:
        top_k = config.custom_llm_api_nl_to_multiple_sql_judge_top_k
    inputs["queries"] = [sql for sql in sqls if sql[1] >= cutoff][:top_k]
    return inputs


async def do_string_literal_correction(inputs: dict) -> dict:
    """
    Do string literal correction on the generated SQL query.
    """
    if not CONFIG.enable_str_literal_correction:
        return inputs
    corrected_queries = []
    for query, score in inputs["queries"]:
        heavydb = await get_db(None)
        corrected_query = await heavydb.acorrect_string_literals(query)
        corrected_queries.append((corrected_query, score))
    inputs["queries"] = corrected_queries
    return inputs


def route(inputs: dict) -> Any:
    """
    Call the judge llm only if the valid SQLs are generated.
    """
    if inputs["valid_sqls"]:
        return (
            RunnablePassthrough.assign(context_str=prepare_judge_llm_context)
            | RunnablePassthrough.assign(judge_llm_output=judge_prompt | judge_llm).with_config(
                config={"tags": ["judge_llm_output"]}
            )
            | RunnableLambda(parse_judge_llm_output_and_assign_scores)  # store judge llm output along with logprobs
        )  # compute the score for each sql query
    return {"queries": [(i, 0.0) for i in inputs["valid_sqls"]]}


multiple_query_chain = (
    SessionRunnable(  # always wrap this chain with SessionRunnable so that the underlying methods may make use of the passed variables
        query_variables
        | RunnablePassthrough.assign(valid_sqls=sql_generator_chain).with_config(
            config={"tags": ["gen_sqls"]}
        )  # filter the queries using sql_validate func
        | RunnableLambda(route)
        | RunnableLambda(apply_top_k_and_cutoff)
        | RunnableLambda(do_string_literal_correction)  # do string literal correction
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoMultipleSQLJudgeChain"], "run_name": "NL to Multiple SQL Judge Chain"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlMultipleChainOutputType)
)  # type: ignore

slim_chain = multiple_query_chain | RunnableLambda(lambda x: x["queries"])
chain = slim_chain
