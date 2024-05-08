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
        end_token=CONFIG.custom_llm_api_rag_prompt_start_token,
    )


REL_EVAL_TEMPLATE = format_prompt(prompt_template=DEFAULT_REL_EVAL_TEMPLATE.template)
REL_REFINE_TEMPLATE = format_prompt(prompt_template=DEFAULT_REL_REFINE_TEMPLATE.template)

TEXT_QA_PROMPT_TMPL = format_prompt(prompt_template=DEFAULT_TEXT_QA_PROMPT_TMPL)
TEXT_QA_PROMPT = PromptTemplate(TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)
