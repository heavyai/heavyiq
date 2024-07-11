import re

from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough

from heavyiq.langchain.heavydb import get_config, get_db
from heavyiq.langchain.llms import LLMType, is_using_custom_trained_llm
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit
from heavyiq.lcel.chains.utils import configure_step, get_value_from_runnable_binding
from heavyiq.lcel.llms import llm_runnable
from heavyiq.lcel.prompts import to_tables_prompt_runnable
from heavyiq.lcel.types import TableChainInputType, TableChainOutputType

CONFIG = get_config()

# llm
nl_to_tables_llm_rbl = llm_runnable.with_config(
    configurable={"llm": "nl_to_tables", "llm_temperature": 0}, config={"tags": ["nl_to_tables_llm"]}  # type: ignore
)
# prompt
nl_to_tables_prompt_rbl = (
    to_tables_prompt_runnable.with_config(configurable={"prompt": "custom"})
    if is_using_custom_trained_llm()
    else to_tables_prompt_runnable
)


async def get_tables(inputs: dict, use_rag: bool = False) -> list[str]:
    """
    Return a list of allowed tables by doing similarity search on chroma db when tables length goes beyond certain limit.
    """
    from heavyrag.main import determine_table_names

    db, config = await get_db(inputs["session_id"]), get_config()

    allowed_tables: list[str] = inputs.get("allowed_tables", [])
    if not allowed_tables:
        allowed_tables = list(db.get_usable_table_names())

    found_tables: list[str] = []

    if (len(allowed_tables) > config.allowed_tables_max_count_nl_to_tables) or use_rag:
        found_tables = await determine_table_names(question=inputs["question"], heavydb=db)

    return found_tables or allowed_tables


async def get_and_retrieve_table_info(inputs: dict) -> str:
    """
    Identify tables and retieve it's relevant information.
    """
    tables_found = await get_tables(inputs, use_rag=False)
    inputs["tables"] = tables_found
    table_info = None

    try:
        table_info = await get_table_info(inputs)
    except RuntimeError:
        # if token exceeds for the previously passed tables
        # then try to reduce the tables list by using the RAG approach
        # this time unnecessary tables get filtered out

        tables = await get_tables(inputs, use_rag=True)
        if tables and (len(tables_found) == len(tables)):
            tables.pop()
        # try to reduce the table on each time until we get the table info
        # without token limit exceeds exception being raised
        while len(tables) >= 1:
            inputs["tables"] = tables
            try:
                table_info = await get_table_info(inputs)
            except RuntimeError:
                tables.pop()
            else:
                break

    if not table_info:
        raise RuntimeError("Couldn't find suitable prompt provided token limit")
    return table_info


async def get_table_info(inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """

    prompt_rbl = nl_to_tables_prompt_rbl
    partial_inputs = {"input": inputs["question"], "relevant_info": inputs.get("relevant_info", "")}

    partial_gen_sql_prompt = get_value_from_runnable_binding(prompt_rbl).partial(**partial_inputs)  # type: ignore
    table_info = await aget_table_info_wrt_token_limit(
        get_value_from_runnable_binding(nl_to_tables_llm_rbl), inputs["session_id"], partial_gen_sql_prompt, inputs["tables"], caller=LLMType.NL_TO_TABLES  # type: ignore
    )
    return table_info


async def parse_output(text: str) -> dict[str, int]:
    """
    Function which parses the final output to return a dict of (table_name: presence_as_int) kv pair.
    """
    return {x: int(y) for x, y in re.findall(r"(\w+):\s*([01])", text)}


# Step 0
async def get_relevant_info_using_rag(inputs: dict) -> str:
    """
    Helps to get the table info for the prompt based upon the allowed token limit.
    """
    from heavyrag.main import get_relevant_facts_info

    if not CONFIG.custom_prompt_nl_to_tables_include_relevant_info:
        return ""

    pre_calculated_relevant_info = inputs.get("pre_calculated_relevant_info", None)
    if pre_calculated_relevant_info is not None:
        return pre_calculated_relevant_info

    heavydb = await get_db(inputs["session_id"])

    relevant_facts = await get_relevant_facts_info(
        question=inputs["question"],
        heavydb_name=heavydb._dbname,
        similarity_cutoff=CONFIG.rag_facts_similarity_cutoff_score,
        similarity_top_k=CONFIG.rag_facts_similarity_top_k,
        with_reranker=True,
        reranker_top_k=CONFIG.rag_facts_reranker_top_k,
        reranker_cutoff=CONFIG.rag_facts_reranker_cutoff_score,
    )

    if not relevant_facts:
        return ""
    # has relevant facts
    return f"Relevant info:\n\n{relevant_facts}\n"


# calculating relevant info (RAG)
relevant_info_lambda: Runnable = RunnableLambda(get_relevant_info_using_rag).with_config(
    config={
        "tags": ["intermediate-step"],
        "run_name": "Get Relevant Info for NLtoTables",
        "metadata": {"step": "Getting relevant information based on the asked question."},
    }
)

# Step 1 and Step 2 combined
get_and_retrieve_table_info_lambda = configure_step(
    RunnableLambda(get_and_retrieve_table_info),
    run_name="Get and Retrieve Table Information",
    step="Identify and Retrieve table information.",
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
        RunnablePassthrough.assign(relevant_info=relevant_info_lambda)
        | RunnablePassthrough.assign(table_info=get_and_retrieve_table_info_lambda, input=lambda x: x["question"])
        | prompt
        | model
        | StrOutputParser()  # needed for chat models to efficiently convert chat message instance to str
        | parse_output_lambda
    )
    .with_config(config={"tags": ["NLtoTablesChainRunnable"], "run_name": "NL to Tables Chain Runnable"})
    .with_types(input_type=TableChainInputType, output_type=TableChainOutputType)  # type: ignore
)

# chain which returns found tables as a list
tables_list_chain = chain | RunnableLambda(lambda x: [k for k, v in x.items() if v == 1])
