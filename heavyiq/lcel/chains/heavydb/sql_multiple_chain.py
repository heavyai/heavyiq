# Generate multiple SQL queries
# Pass them to judge llm which is supposed to produce 1 and 0 for valid sql queries
# Pick the right one having the highest probability
import math
from typing import Any

from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.language_models.llms import LLMResult
from langchain_core.outputs.generation import Generation

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import multiple_sql_judge_prompt, to_sql_prompt_runnable
from heavyiq.lcel.types import SqlChainInputType, SqlMultipleChainOutputType

from .sql_chain import table_info_runnable_lambda

CONFIG = get_config()

llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_multiple_sql", "llm_temperature": 0.9, "llm_n": 5}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
)
judge_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_multiple_sql_judge", "llm_temperature": 0.0, "llm_n": 1}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
)
prompt_rbl = (
    to_sql_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_sql_prompt_runnable
)

# Step 1
query_variables = (RunnablePassthrough.assign(table_info=table_info_runnable_lambda, input=lambda x: x["question"])).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calculate Input Variables",
        "metadata": {"step": "Calculating input variables for Query prompt."},
    }
)

# Step 2
query_prompt: Runnable = prompt_rbl.with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Preparing NL-Multiple-SQL Prompt",
        "metadata": {"step": "Constructing prompt with input variables."},
    }
)


# Create a RunnableLambda that wraps the llm.invoke call
class LLMGeneratorRunnableLambda(Runnable):
    """
    Custom runnable lambda which accepts llm runnable.
    Always call the parent runnable (ie.runnbale which includes this Runnable) with invoke and ainvoke methods.
    """

    def __init__(
        self,
        llm_runnable: Runnable,
        name: str | None = None,
    ) -> None:
        self.llm_runnable = llm_runnable
        self.name = name or "LLMGeneratorRunnable"

    async def _ainvoke(self, input_: Any, *args, **kwargs: Any) -> list[Generation]:
        """
        Custom ainvoke method.
        """
        # Call the invoke method of the language model
        llm = get_value_from_runnable_binding(self.llm_runnable)
        result: LLMResult = await llm.agenerate_prompt([input_], **kwargs)
        generations = []
        for gen in result.generations[0]:
            generations.append(gen)
        return generations

    def _invoke(self, input_: Any, *args, **kwargs: Any) -> list[Generation]:
        llm = get_value_from_runnable_binding(llm_rbl)
        result: LLMResult = llm.generate_prompt([input_], **kwargs)
        generations = []
        for gen in result.generations[0]:
            generations.append(gen)
        return generations

    async def ainvoke(self, input_: Any, *args, **kwargs: Any) -> list[tuple[str, dict]]:
        """
        Custom ainvoke method which helps to generate n answers.
        """
        # Extract the full OpenAI response
        generations = await self._ainvoke(input_, *args, **kwargs)
        return [(i.text, i.generation_info["logprobs"]) for i in generations]

    def invoke(self, input_: Any, *args, **kwargs: Any | None) -> list[tuple[str, dict]]:
        generations = self._invoke(input_, *args, **kwargs)
        return [(i.text, i.generation_info["logprobs"]) for i in generations]


# Step 3
query_llm: Runnable = LLMGeneratorRunnableLambda(llm_runnable=llm_rbl).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calling Generate SQL LLM",
        "metadata": {"step": "Invoking the LLM for generating mutiple SQL queries."},
    }
)

judge_llm: Runnable = LLMGeneratorRunnableLambda(llm_runnable=judge_llm_rbl).with_config(
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


def get_gen_llm_text(llm_output: tuple) -> list[str]:
    """
    Get only the text from gen llm output.
    """
    return [i[0] for i in llm_output]


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


def parse_judge_llm_output_and_assign_scores(inputs: dict) -> list[str, float]:
    """
    Helps to assign SQL scores based on log probability.
    """
    valid_sqls = inputs["valid_sqls"]
    _, logprobs = inputs["judge_llm_output"][0]
    scores = find_probability_score_from_logprobs(logprobs)
    assert len(valid_sqls) == len(scores)
    return list(zip(valid_sqls, scores))


def filter_queries_by_threshold(sqls: list[str], threshold: float = 0.5):
    """
    Filters the sqls in the list that meet or exceed the threshold value.
    """
    return [sql[0] for sql in sqls if sqls[1] >= threshold]


chain = (
    (
        RunnablePassthrough.assign(relevant_info=RunnableLambda(lambda x: ""))
        | query_variables
        | RunnablePassthrough.assign(
            sqls=query_prompt | query_llm | RunnableLambda(get_gen_llm_text)
        )  # generates n SQL queries
        | RunnablePassthrough.assign(
            valid_sqls=RunnableLambda(filter_valid_queries)
        )  # filter the queries using sql_validate func
        | RunnablePassthrough.assign(context_str=prepare_judge_llm_context)  # build prompt varibales for judge llm
        | RunnablePassthrough.assign(
            judge_llm_output=multiple_sql_judge_prompt | judge_llm
        )  # store judge llm output along with logprobs
        | RunnableLambda(parse_judge_llm_output_and_assign_scores)  # compute the score for each sql query
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoMultipleSQLJudgeChain"], "run_name": "NL to Multiple SQL Judge Chain"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType, output_type=SqlMultipleChainOutputType)
)  # type: ignore
