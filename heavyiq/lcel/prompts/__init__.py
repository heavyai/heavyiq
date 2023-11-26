from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import ConfigurableField

from heavyiq.langchain.chains.heavydb.nl_to_answer import ANSWER_TEMPLATE, CUSTOM_LLM_ANSWER_TEMPLATE
from heavyiq.langchain.chains.heavydb.nl_to_sql import CUSTOM_LLM_NL_TO_SQL_TEMPLATE as CUSTOM_NL_TO_SQL_TEMPLATE
from heavyiq.langchain.chains.heavydb.nl_to_sql import (
    CUSTOM_NL_TO_SQL_ERROR_TEMPLATE,
    NL_TO_SQL_ERROR_TEMPLATE,
    NL_TO_SQL_TEMPLATE,
)
from heavyiq.langchain.chains.heavydb.nl_to_tables import NL_TO_TABLES_CUSTOM_TEMPLATE, NL_TO_TABLES_TEMPLATE

to_sql_prompt_runnable = PromptTemplate.from_template(NL_TO_SQL_TEMPLATE).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="prompt"),
    # This sets a default_key.
    # If we specify this key, the default NL_TO_SQL_TEMPLATE will be used
    default_key="openai",
    # This adds a new option, with name `custom` which defines the custom prompt
    custom=PromptTemplate.from_template(CUSTOM_NL_TO_SQL_TEMPLATE),
    # This adds a new option, with name `openai_error` which defines the openai prompt for error/retrying purpose.
    openai_error=PromptTemplate.from_template(NL_TO_SQL_ERROR_TEMPLATE),
    # This adds a new option, with name `custom_error` which defines the custom prompt for error/retrying purpose.
    custom_error=PromptTemplate.from_template(CUSTOM_NL_TO_SQL_ERROR_TEMPLATE),
)
to_answer_prompt_runnable = PromptTemplate.from_template(ANSWER_TEMPLATE).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="openai",
    custom=PromptTemplate.from_template(CUSTOM_LLM_ANSWER_TEMPLATE),
)

to_tables_prompt_runnable = PromptTemplate.from_template(NL_TO_TABLES_TEMPLATE).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="openai",
    custom=PromptTemplate.from_template(NL_TO_TABLES_CUSTOM_TEMPLATE),
)
