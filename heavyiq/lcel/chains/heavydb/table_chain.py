import re

from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.index import aget_heavydb_index
from heavyiq.langchain.llms import is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import configure_step, get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_tables_prompt_runnable
from heavyiq.lcel.types import TableChainInputType, TableChainOutputType

# llm
nl_to_tables_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "default_llm", "llm_temperature": 0}, config={"tags": ["nl_to_tables_llm"]}  # type: ignore
)
# prompt
nl_to_tables_prompt_rbl = (
    to_tables_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_tables_prompt_runnable
)


async def get_tables(inputs: dict) -> list[str]:
    """
    Return a list of tables by doing similarity search on chroma db.
    """
    db = await get_db(inputs["session_id"])
    all_tables: set[str] = db.get_usable_table_names()

    if len(all_tables) > 10:
        heavydb_index = await aget_heavydb_index()
        found_tables = set(
            heavydb_index.simple_search_for_table_names(inputs["question"], allowable_tables=list(all_tables))
        )

    return list(found_tables or all_tables)


async def get_table_info(inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """

    prompt_rbl = nl_to_tables_prompt_rbl
    partial_inputs = {"input": inputs["question"]}

    heavydb = await get_db(inputs["session_id"])
    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl).partial(**partial_inputs)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_tables_llm_rbl), heavydb, partial_gen_sql_prompt, inputs["tables"]  # type: ignore
    )
    return table_info


async def parse_output(text: str) -> dict[str, int]:
    """
    Function which parses the final output to return a dict of (table_name: presence_as_int) kv pair.
    """
    return {x: int(y) for x, y in re.findall(r"(\w+):\s*([01])", text)}


# Step 1
# Find relevant tables by doing smilarity search on chroma db
get_tables_lambda = configure_step(
    RunnableLambda(get_tables),
    run_name="Similarity Search for Relevant Tables",
    step="Finding relevant tables by querying ChromaDB.",
)

# Step 2
# Retrieve table information for the identified tables in order to construct the prompt
retrieve_tables_info_lambda = configure_step(
    RunnableLambda(get_table_info),
    run_name="Retrieve Table Information",
    step="Retrieving table information for the identified tables.",
)

# Step 3
# Prompt formatting
prompt = configure_step(nl_to_tables_prompt_rbl, run_name="Format Prompt", step="Formatting prompt.")

# Step 4
# Call LLM
model = configure_step(nl_to_tables_llm_rbl, run_name="Call LLM", step="Calling LLM.")

# Step 5
# Parse the final output
parse_output_lambda = configure_step(
    RunnableLambda(parse_output), run_name="Parse LLM Output", step="Parsing LLM Output."
)

chain: Runnable = (
    (
        RunnablePassthrough.assign(tables=get_tables_lambda)
        | RunnablePassthrough.assign(table_info=retrieve_tables_info_lambda, input=lambda x: x["question"])
        | prompt
        | model
        | parse_output_lambda
    )
    .with_config(config={"tags": ["NLtoTablesChainRunnable"], "run_name": "NL to Tables Chain Runnable"})
    .with_types(input_type=TableChainInputType, output_type=TableChainOutputType)  # type: ignore
)

# chain which returns found tables as a list
tables_list_chain = chain | RunnableLambda(lambda x: [k for k, v in x.items() if v == 1])
