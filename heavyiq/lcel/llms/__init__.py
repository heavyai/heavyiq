import re
from functools import partial
from typing import Any

from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.prompts.base import StringPromptValue
from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable import ConfigurableField, RunnableBinding, RunnableConfig
from langchain.schema.runnable.configurable import RunnableConfigurableAlternatives

from heavyiq.config import get_config
from heavyiq.config.config_schema import HeavyIQConfig
from heavyiq.langchain.llms import LLMType, get_llm_by_type

global_config: HeavyIQConfig = get_config()


_llm_runnable = (
    OpenAI(temperature=0, openai_api_key="nothing")
    .configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="openai",
        nl_to_sql=partial(get_llm_by_type, LLMType.NL_TO_SQL, temperature=0.0),  # type: ignore
        sql_to_answer=partial(get_llm_by_type, LLMType.SQL_TO_ANSWER, temperature=0.0),  # type: ignore
        nl_to_tables=partial(get_llm_by_type, LLMType.NL_TO_TABLES, temperature=0.0),  # type: ignore
        default_llm=partial(get_llm_by_type, LLMType.DEFAULT, temperature=0.0),  # type: ignore
    )
    .configurable_fields(
        temperature=ConfigurableField(
            id="llm_temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )
)


class LLMRunnableBinding(RunnableBinding):
    async def ainvoke(self, input: StringPromptValue, config: RunnableConfig | None = None, **kwargs) -> Any:
        # check nl_to_sql llm call or not
        if (
            self.config["configurable"]["llm"] in ["nl_to_sql", "nl_to_tables"]
            and global_config.enable_vllm_prefix_cache
            and global_config.custom_llm_type == "API_VLLM"
        ):
            # check for prefix caching enabled or not
            text = input.to_string()
            # split input prompt value to prefix and question
            prefix, delimiter, completion = re.split(r"(?i)(?<=question:)(\s)", text)
            return await super().ainvoke(StringPromptValue(text=completion), config=config, prefix=prefix + delimiter)

        return await super().ainvoke(input, config=config, **kwargs)


llm_runnable = LLMRunnableBinding(bound=_llm_runnable)
