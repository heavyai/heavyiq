from langchain.llms.openai import OpenAI
from langchain.schema.runnable import ConfigurableField

from heavyiq.langchain.llms import LLMType, get_llm_by_type

llm_runnable = (
    OpenAI(temperature=0, openai_api_key="nothing")
    .configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="openai",
        nl_to_sql=get_llm_by_type(LLMType.NL_TO_SQL, temperature=0.0),  # type: ignore
        sql_to_answer=get_llm_by_type(LLMType.SQL_TO_ANSWER, temperature=0.0),  # type: ignore
        nl_to_tables=get_llm_by_type(LLMType.NL_TO_TABLES, temperature=0.0),  # type: ignore
        default_llm=get_llm_by_type(LLMType.DEFAULT, temperature=0.0),  # type: ignore
    )
    .configurable_fields(
        temperature=ConfigurableField(
            id="llm_temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )
)
