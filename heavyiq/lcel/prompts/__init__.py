from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import ConfigurableField

from heavyiq.langchain.llms import LLMType
from heavyiq.lcel.prompts.base import get_prompt_by_llm_type
from heavyiq.lcel.prompts.openai import (
    ANSWER_TEMPLATE,
    NL_TO_SQL_ERROR_TEMPLATE,
    NL_TO_SQL_TEMPLATE,
    NL_TO_TABLES_TEMPLATE,
)
from heavyiq.lcel.prompts.question_prompt import TABLES_TO_NL_QUESTIONS_TEMPLATE

CUSTOM_NL_TO_SQL_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL)
CUSTOM_NL_TO_SQL_ERROR_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL_ERROR)
CUSTOM_LLM_ANSWER_TEMPLATE = get_prompt_by_llm_type(LLMType.SQL_TO_ANSWER)
NL_TO_TABLES_CUSTOM_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_TABLES)
TABLES_TO_NL_QUESTIONS_CUSTOM_TEMPLATE = get_prompt_by_llm_type(LLMType.TABLES_TO_QUESTIONS)
CUSTOM_NL_TO_SQL_COT_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL_COT, prompt_type="custom")
NL_TO_SQL_COT_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL_COT, prompt_type="openai")
NL_TO_SQL_COT_ERROR_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL_COT_ERROR, prompt_type="openai")
CUSTOM_NL_TO_SQL_COT_ERROR_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_SQL_COT_ERROR, prompt_type="custom")
CUSTOM_NL_TO_MULTIPLE_SQL_JUDGE_TEMPLATE = get_prompt_by_llm_type(
    LLMType.NL_TO_MULTIPLE_SQL_JUDGE, prompt_type="custom"
)
CUSTOM_NL_TO_VEGA_LITE_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_VEGA_LITE, prompt_type="custom")
CUSTOM_VEGA_LITE_ERROR_TEMPLATE = get_prompt_by_llm_type(LLMType.NL_TO_VEGA_LITE_ERROR, prompt_type="custom")

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

to_questions_prompt_runnable = PromptTemplate.from_template(TABLES_TO_NL_QUESTIONS_TEMPLATE).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="openai",
    custom=PromptTemplate.from_template(TABLES_TO_NL_QUESTIONS_CUSTOM_TEMPLATE),
)

to_sql_with_cot_prompt_runnable = PromptTemplate.from_template(NL_TO_SQL_COT_TEMPLATE).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="openai",
    custom=PromptTemplate.from_template(CUSTOM_NL_TO_SQL_COT_TEMPLATE),
    openai_error=PromptTemplate.from_template(NL_TO_SQL_COT_ERROR_TEMPLATE),
    custom_error=PromptTemplate.from_template(CUSTOM_NL_TO_SQL_COT_ERROR_TEMPLATE),
)

multiple_sql_judge_prompt = PromptTemplate.from_template(CUSTOM_NL_TO_MULTIPLE_SQL_JUDGE_TEMPLATE)

to_vega_lite_prompt_runnable = PromptTemplate.from_template(CUSTOM_NL_TO_VEGA_LITE_TEMPLATE)
correct_vega_lite_prompt_runnable = PromptTemplate.from_template(CUSTOM_VEGA_LITE_ERROR_TEMPLATE)
