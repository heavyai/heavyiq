# Gen command chain to generate Chain of Thoughts for the given question and gold query
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField, RunnableLambda, RunnableParallel, RunnablePassthrough
from typing_extensions import Any

from heavyiq.config import get_config
from heavyiq.langchain.utils import aget_table_info_wrt_token_limit

raw_prompt = """
From the given natural language question and gold sql query, return the relevant Chain Of Thoughts reasoning back as an ordered list.

Question: [QUESTION]
SQLQuery: [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.

{table_info}
"""
config = get_config()

# configurable model field where it's value can be passed from the invoke call
chat_openai = ChatOpenAI(model_name=config.openai_gpt_model, openai_api_key=config.openai_api_key).configurable_fields(  # type: ignore
    model_name=ConfigurableField(
        id="llm_model",
        name="LLM Model",
        description="The model to be used for the LLM",
    )
)


async def get_prompt(inputs: dict[str, Any]) -> ChatPromptTemplate:
    """
    Gets the prompt.
    """
    heavydb, tables, question, query = (
        inputs.get("heavydb"),
        inputs.get("tables"),
        inputs.get("question"),
        inputs.get("query"),
    )
    # build prompt from messages
    messages = [("system", raw_prompt), ("human", "\nQuestion: {question}\nSQLQuery: {query}")]
    prompt = ChatPromptTemplate.from_messages(messages)
    partial_prompt = prompt.partial(question=question, query=query)

    table_info = await aget_table_info_wrt_token_limit(
        llm=ChatOpenAI(openai_api_key=config.openai_api_key),
        heavydb=heavydb,
        prompt=partial_prompt,
        table_names_to_use=tables,
    )

    return partial_prompt.partial(table_info=table_info)


async def format_output(inputs: dict) -> dict:
    """
    Formats the final output.
    """
    chain_inputs = inputs["inputs"].copy()
    heavydb = chain_inputs.pop("heavydb")
    return {**chain_inputs, "cot": inputs["cot"], "db_id": heavydb._dbname}


chain = (
    RunnableParallel(
        cot=(
            {
                "question": lambda x: x["question"],
                "query": lambda x: x["query"],
                "heavydb": lambda x: x["heavydb"],
                "tables": lambda x: x["tables"],
            }
            | RunnableLambda(get_prompt)  # type: ignore
            | chat_openai
            | StrOutputParser()
        ),
        inputs=RunnablePassthrough(),
    )
    | format_output
)
