from llama_index.core.evaluation.relevancy import DEFAULT_EVAL_TEMPLATE as DEFAULT_REL_EVAL_TEMPLATE
from llama_index.core.evaluation.relevancy import DEFAULT_REFINE_TEMPLATE as DEFAULT_REL_REFINE_TEMPLATE
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from llama_index.core.prompts.prompt_type import PromptType

from heavyiq.config import get_config

CONFIG = get_config()
RAG_TEMPLATE = "{start_token}{user_message}{end_token}"


def format_prompt(prompt_template: str) -> str:
    """
    Formats the RAG prompt by prepending and appending start and end tokens.
    """
    return RAG_TEMPLATE.format(
        start_token=CONFIG.custom_llm_api_rag_prompt_start_token,
        user_message=prompt_template,
        end_token=CONFIG.custom_llm_api_rag_prompt_end_token,
    )


REL_EVAL_TEMPLATE = format_prompt(prompt_template=DEFAULT_REL_EVAL_TEMPLATE.template)
REL_REFINE_TEMPLATE = format_prompt(prompt_template=DEFAULT_REL_REFINE_TEMPLATE.template)

TEXT_QA_PROMPT_TMPL = format_prompt(prompt_template=DEFAULT_TEXT_QA_PROMPT_TMPL)
TEXT_QA_PROMPT = PromptTemplate(TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)


FACTS_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "Retrieve facts relevant to the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_REFINE_TABLE_NAME_TMPL = (
    "We have provided multiple table schemas with optional sample data below. "
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "We have also provided the relevant tables found below. "
    "{existing_answer}\n"
    "---------------------\n"
    "Given the table schemas and the list of tables,"
    "you have to pick the possible tables from the above tables list"
    "which are likely to generate results for the input question: {query_str}\n"
    "Provide the final list of tables in comma seprated format."
)
REFINE_TABLE_NAME_PROMPT = PromptTemplate(
    format_prompt(DEFAULT_REFINE_TABLE_NAME_TMPL), prompt_type=PromptType.TABLE_CONTEXT
)
FACTS_QA_PROMPT = PromptTemplate(format_prompt(FACTS_PROMPT_TMPL), prompt_type=PromptType.QUESTION_ANSWER)
