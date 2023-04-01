"""HeavyDB agent."""
from typing import Any, Optional

from langchain.agents.agent import AgentExecutor
from langchain.agents.chat.base import ChatAgent
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS as CHAT_FORMAT_INSTRUCTIONS
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel

from modules.langchain.agents.agent_toolkits.heavydb.toolkit import HeavyDBToolkit
from modules.langchain.agents.agent_toolkits.heavydb.prompts import SQL_PREFIX, SQL_SUFFIX


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
    """Construct a HeavyDB agent from an LLM and tools."""
    tools = toolkit.get_tools()
    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)
    tool_names = [tool.name for tool in tools]

    if isinstance(llm, BaseChatModel):
        prompt = ChatAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=CHAT_FORMAT_INSTRUCTIONS,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        agent = ChatAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    else:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=toolkit.get_tools(), verbose=verbose)
