# Generate multiple SQL queries
# Pass them to judge llm which is supposed to produce 1 and 0 for valid sql queries
# Pick the right one having the highest probability
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

from langchain.schema.runnable import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.language_models.llms import LLMResult
from langchain_core.outputs.generation import Generation
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.lcel.chains.utils import get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import multiple_sql_judge_prompt, to_sql_prompt_runnable

from .sql_chain import SqlChainInputType, table_info_runnable_lambda

CONFIG = get_config()

llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_multiple_sql", "llm_temperature": 0.9, "llm_n": 5}, config={"tags": ["nl_to_sql_llm"]}  # type: ignore
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

    async def ainvoke(self, input_: Any, *args, **kwargs: Any) -> dict:
        """
        Custom ainvoke method which helps to generate n answers.
        """
        # Extract the full OpenAI response
        generations = await self._ainvoke(input_, *args, **kwargs)
        return [i.text for i in generations]

    def invoke(self, input_: Any, *args, **kwargs: Any | None) -> list[str]:
        generations = self._invoke(input_, *args, **kwargs)
        return [i.text for i in generations]


# Step 3
query_llm: Runnable = LLMGeneratorRunnableLambda(llm_runnable=llm_rbl).with_config(  # type: ignore
    config={
        "tags": ["intermediate-step"],
        "run_name": "Calling LLM",
        "metadata": {"step": "Invoking the LLM for generating mutiple SQL queries."},
    }
)


def parse_inputs(inputs):
    print(inputs)
    return inputs


chain = (
    (
        RunnablePassthrough.assign(relevant_info=RunnableLambda(lambda x: ""))
        | query_variables
        | RunnablePassthrough.assign(sqls=query_prompt | query_llm)
        | RunnableLambda(parse_inputs)
    )
    .with_config(  # type: ignore
        config={"tags": ["NLtoSQLChainRunnable"], "run_name": "NL to SQL Chain Runnable"}  # type: ignore
    )
    .with_types(input_type=SqlChainInputType)
)  # type: ignore
