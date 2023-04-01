"""HeavyDB agent."""
from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.chat.base import ChatAgent
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS as CHAT_FORMAT_INSTRUCTIONS
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool

from modules.langchain.agents.agent_toolkits.heavydb.toolkit import HeavyDBToolkit
from modules.langchain.agents.agent_toolkits.heavydb.prompts import SQL_PREFIX, SQL_SUFFIX


def create_llm_chain(
    llm: BaseLLM | BaseChatModel, prompt: str, callback_manager: Optional[BaseCallbackManager] = None
) -> LLMChain:
    return LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )


def create_agent(
    llm: BaseLLM | BaseChatModel, llm_chain: LLMChain, tools: list[BaseTool], **kwargs: Any
) -> AgentExecutor:
    allowed_tools = [tool.name for tool in tools]

    if isinstance(llm, BaseChatModel):
        return ChatAgent(llm_chain=llm_chain, allowed_tools=allowed_tools, **kwargs)
    else:
        return ZeroShotAgent(llm_chain=llm_chain, allowed_tools=allowed_tools, **kwargs)


def create_heavydb_agent(
    llm: BaseLLM | BaseChatModel,
    toolkit: HeavyDBToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    input_variables: Optional[list[str]] = None,
    top_k: int = 10,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    tools = toolkit.get_tools()
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)

    if isinstance(llm, BaseChatModel):
        format_instructions = CHAT_FORMAT_INSTRUCTIONS
    else:
        format_instructions = FORMAT_INSTRUCTIONS

    prompt = ChatAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )

    llm_chain = create_llm_chain(llm=llm, prompt=prompt, callback_manager=callback_manager)

    return AgentExecutor.from_agent_and_tools(
        agent=create_agent(llm, llm_chain, tools, **kwargs),
        tools=toolkit.get_tools(),
        verbose=verbose,
    )
