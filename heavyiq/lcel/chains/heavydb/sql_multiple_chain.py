# Generate multiple SQL queries
# Pass them to judge llm which is supposed to produce 1 and 0 for valid sql queries
# Pick the right one having the highest probability
import math
from typing import Any

from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.language_models.llms import LLMResult
from langchain_core.outputs.generation import Generation

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import LLMType, get_llm_by_type, is_using_custom_trained_llm
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import multiple_sql_judge_prompt, to_sql_prompt_runnable

# from .sql_chain import table_info_runnable_lambda
from heavyiq.lcel.runnables.base import LLMGeneratorRunnable, SessionRunnable
from heavyiq.lcel.types import SqlChainInputType, SqlMultipleChainOutputType, SQLwithScore

from .sql_gen_chain import filter_valid_queries_chain as sql_generator_chain
from .sql_gen_chain import gen_query_variables as query_variables

CONFIG = get_config()


# judge_llm_rbl = llm_runnable.with_config(
#     configurable={"llm": "nl_to_multiple_sql_judge", "llm_temperature": 0.0, "llm_n": 1}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
# )
judge_llm_rbl = get_llm_by_type(LLMType.NL_TO_MULTIPLE_SQL_JUDGE, temperature=0.0)

judge_llm: Runnable = LLMGeneratorRunnable(llm_runnable=judge_llm_rbl, name="sql-judge-llm").with_config(
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calling Judge LLM",
        "metadata": {"step": "Invoking the LLM for judging mutiple SQL queries."},
    }
)


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


def route(inputs: dict) -> Any:
    """
    Call the judge llm only if the valid SQLs are generated.
    """
    if inputs["valid_sqls"]:
        return (
            RunnablePassthrough.assign(context_str=prepare_judge_llm_context)
            | RunnablePassthrough.assign(judge_llm_output=multiple_sql_judge_prompt | judge_llm)
            | RunnableLambda(parse_judge_llm_output_and_assign_scores)  # store judge llm output along with logprobs
            | RunnableLambda(apply_top_k_and_cutoff)
        )  # compute the score for each sql query
    return apply_top_k_and_cutoff({"queries": [(i, 0.0) for i in inputs["valid_sqls"]]})


chain = (
    SessionRunnable(
        query_variables
        | RunnablePassthrough.assign(valid_sqls=sql_generator_chain)  # filter the queries using sql_validate func
        | RunnableLambda(route)
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoMultipleSQLJudgeChain"], "run_name": "NL to Multiple SQL Judge Chain"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlMultipleChainOutputType)
)  # type: ignore

slim_chain = chain | RunnableLambda(lambda x: x["queries"])
