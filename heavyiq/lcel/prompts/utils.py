from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from heavyiq.lcel.chains.utils import get_value_from_runnable_binding


def get_prompt_template_from_prompt_runnable(prompt_runnable: Runnable, **kwargs) -> PromptTemplate:
    """
    Helps to get the prompt template from prompt runnables.
    **kwargs: additonal prompt input variables
    """
    prompt: PromptTemplate = get_value_from_runnable_binding(prompt_runnable)
    return prompt.partial(**prompt_runnable.kwargs, **kwargs)
