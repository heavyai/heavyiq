from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.llms import LLMType, is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import configure_step, get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_questions_prompt_runnable
from heavyiq.lcel.types import QuestionsChainInputType, QuestionsChainInputTypedDict, QuestionsChainOutputType

# llm
tables_to_questions_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_tables", "llm_temperature": 0}, config={"tags": ["nl_to_tables_llm"]}  # type: ignore
)
# prompt
tables_to_questions_prompt_rbl = (
    to_questions_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_questions_prompt_runnable
)


async def get_table_info(inputs: QuestionsChainInputTypedDict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """

    prompt_rbl = tables_to_questions_prompt_rbl

    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(tables_to_questions_llm_rbl), inputs["session_id"], partial_gen_sql_prompt, inputs["tables"], caller=LLMType.NL_TO_TABLES  # type: ignore
    )
    return table_info


# Step 1
# Retrieve tables information from HeavyDB
retrieve_tables_info_lambda = configure_step(
    RunnableLambda(get_table_info),
    run_name="Retrieve Table Information",
    step="Retrieving table information for the given tables.",
)

# Step 2
# Prompt formatting
prompt = configure_step(tables_to_questions_prompt_rbl, run_name="Format Prompt", step="Formatting prompt.")

# Step 3
# Call LLM
model = configure_step(tables_to_questions_llm_rbl, run_name="Call LLM", step="Calling LLM.")

chain: Runnable = (
    (RunnablePassthrough.assign(table_info=retrieve_tables_info_lambda) | prompt | model)
    .with_config(config={"tags": ["TablesToQuestionsChain"], "run_name": "Tables to NL Questions Chain"})
    .with_types(input_type=QuestionsChainInputType, output_type=QuestionsChainOutputType)  # type: ignore
)
