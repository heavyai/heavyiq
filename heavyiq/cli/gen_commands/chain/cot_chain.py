# Gen command chain to generate Chain of Thoughts for the given question and gold query
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField, RunnableLambda, RunnableParallel, RunnablePassthrough
from typing_extensions import Any

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.llms import LLMType, get_llm_by_type
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
chat_openai = ChatOpenAI(model_name=config.openai_gpt_model, openai_api_key=config.openai_api_key or "xxxxxxxxx").configurable_fields(  # type: ignore
    model_name=ConfigurableField(
        id="llm_model",
        name="LLM Model",
        description="The model to be used for the LLM",
    )
)

llm = get_llm_by_type(LLMType.DEFAULT)

custom_prompt = """From the given natural language question and gold sql query, return the relevant Chain Of Thoughts reasoning back as an ordered list.

Question: [QUESTION]
SQLQuery: [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.

{table_info}

Question: {question}
SQLQuery: {query}
ASSISTANT: """


async def get_prompt(inputs: dict[str, Any]) -> ChatPromptTemplate | PromptTemplate:
    """
    Gets the prompt.
    """
    heavydb = None
    if db := inputs.get("heavydb"):
        heavydb = db
    else:
        heavydb = await HeavyDB.from_session_async(inputs["session"])
    tables, question, query = (
        inputs.get("tables"),
        inputs.get("question"),
        inputs.get("query"),
    )
    if not config.custom_llm_type:
        # build prompt from messages
        # openai prompt
        messages = [("system", raw_prompt), ("human", "\nQuestion: {question}\nSQLQuery: {query}")]
        prompt = ChatPromptTemplate.from_messages(messages)
        partial_prompt = prompt.partial(question=question, query=query)
    else:
        # custom prompt
        prompt = PromptTemplate.from_template(custom_prompt)
        partial_prompt = prompt.partial(question=question, query=query)

    table_info = await aget_table_info_wrt_token_limit(
        llm=llm,
        session=heavydb._conn._session,
        prompt=partial_prompt,
        table_names_to_use=tables,
    )

    return partial_prompt.partial(table_info=table_info)


async def format_output(inputs: dict) -> dict:
    """
    Formats the final output.
    """
    print(inputs)
    chain_inputs = inputs["inputs"].copy()
    return {**chain_inputs, "cot": inputs["cot"]}


async def get_llm(inputs: dict):
    if not config.custom_llm_type:
        return chat_openai
    return llm


chain = (
    RunnableParallel(
        cot=(
            {
                "question": lambda x: x["question"],
                "query": lambda x: x["query"],
                "session": lambda x: x.get("session"),
                "heavydb": lambda x: x.get("heavydb"),
                "tables": lambda x: x["tables"],
            }
            | RunnableLambda(get_prompt)  # type: ignore
            | RunnableLambda(get_llm)
            | StrOutputParser()
        ),
        inputs=RunnablePassthrough(),
    )
    | format_output
)
