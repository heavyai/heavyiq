from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.response_synthesizers.type import ResponseMode

from heavyrag.common.llm import OverrideVllmServer
from heavyrag.settings import settings

# NOTE: we add an extra tone_name variable here
qa_prompt_tmpl = (
    "<|prompt|>\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Do not give me an answer if it is not mentioned in the context as a fact. \n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: \n"
    "<|answer|>\n"
)
qa_prompt = PromptTemplate(qa_prompt_tmpl, prompt_type=PromptType.QUESTION_ANSWER)

DEFAULT_REFINE_PROMPT_TMPL = (
    "<|prompt|>\n"
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. "
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
    "<|answer|>\n"
)
REFINE_PROMPT = PromptTemplate(DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE)

llm = OverrideVllmServer(
    model=settings.llm_model,
    api_url=settings.llm_api_url,
    max_new_tokens=settings.llm_max_tokens,
    temperature=settings.llm_temperature,
)

compact_refine_response_synthesizer = get_response_synthesizer(
    llm=llm,
    structured_answer_filtering=True,
    text_qa_template=qa_prompt,
    refine_template=REFINE_PROMPT,
    response_mode=ResponseMode.COMPACT,
)

compact_accumulate_response_synthesizer = get_response_synthesizer(
    llm=llm,
    structured_answer_filtering=True,
    text_qa_template=qa_prompt,
    refine_template=REFINE_PROMPT,
    response_mode=ResponseMode.COMPACT_ACCUMULATE,
)

response_synthesizer = compact_accumulate_response_synthesizer
