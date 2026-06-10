# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# TODO: remove this prompt
DEFAULT_RAG_RERANK_CHOICE_SELECT_PROMPT_TMPL = (
    f"{CONFIG.rag_rerank_llm_start_token}A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n"
    "Question: {query_str}\n"
    f"Answer:{CONFIG.rag_rerank_llm_end_token}"
)
RAG_RERANK_CHOICE_SELECT_PROMPT_TMPL = (
    f"{CONFIG.rag_rerank_llm_start_token}A numbered list of potentially relevant hints to answer a question is shown below. The question is also provided. \n"
    "Respond with a list of each hint number along with a 1 if the hint is helpful to answer the question, or a 0 if not.\n"
    "Example format: \n"
    "Hint 1:\n<content of hint 1>\n\n"
    "Hint 2:\n<content of hint 2>\n\n"
    "...\n\n"
    "Question: <question>\n"
    f"{CONFIG.rag_rerank_llm_end_token}\n"
    "Hint 1: 0|1\n"
    "Hint 2: 0|1\n"
    "...\n\n"
    "Let's try this now: \n\n"
    "{context_str}\n\n"
    "Question: {query_str}\n"
    f"{CONFIG.rag_rerank_llm_end_token}"
)
DEFAULT_RAG_RERANK_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_RAG_RERANK_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)
RAG_RERANK_CHOICE_SELECT_PROMPT = PromptTemplate(
    RAG_RERANK_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)
