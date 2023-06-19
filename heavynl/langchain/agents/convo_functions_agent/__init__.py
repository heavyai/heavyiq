from typing import Any, List, Optional, Tuple, Union

from langchain.agents import AgentExecutor, AgentType
from langchain.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    _format_intermediate_steps,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.callbacks.manager import Callbacks
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AgentAction,
    AgentFinish,
    SystemMessage,
)
from langchain.agents import initialize_agent

from heavynl.config import get_config
from heavynl.langchain import HeavyDB
from heavynl.langchain.agents import HeavyDBToolkit
from heavynl.langchain.llms import get_chat_llm
from heavynl.langchain.agents.messages import (
    SYSTEM_MESSAGE,
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE_WITHOUT_INPUT,
)


def create_conversational_function_agent(
    heavydb: Optional[HeavyDB] = None, chat_llm: Optional[BaseChatModel] = None, verbose: bool = False
) -> AgentExecutor:
    """This function creates an AgentExecutor instance that uses a language model to generate responses
    based on user inputs. This function uses the newly introduced openai's function call feature,
    where it automatically picks the right tool and calls it with exact arguments.
    The agent is designed to interact with a HeavyDB instance and has access
    to a set of tools for querying the database. The agent follows a specific prompt template and
    returns the final answer after executing the necessary SQL queries or directly responds to the
    user if no tool is required. The language model has a memory of the conversation.

    Parameters:
    - heavydb (Optional[HeavyDB]): A HeavyDB instance to interact with. If not provided, a default
    HeavyDB instance will be created using environment variables.
    - chat_llm (Optional[BaseChatModel], optional): A language model instance to use for generating
    responses. If not provided, a default ChatOpenAI instance with GPT-4 will be used.
    - verbose (bool, optional): If True, the AgentExecutor will print additional information during
    execution. Defaults to False.

    Returns:
    - AgentExecutor: An AgentExecutor instance configured with the conversational agent and the set
    of tools for interacting with the database."""

    chat_llm = chat_llm or get_chat_llm(
        ["agent", "conversational_sql_agent"], temperature=0, model=get_config().openai_gpt_model
    )
    heavydb = heavydb or HeavyDB.from_env()
    toolkit = HeavyDBToolkit(db=heavydb)
    tools = toolkit.get_new_tools()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, ai_prefix="Assistant")

    tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = HUMAN_MESSAGE_WITHOUT_INPUT.format(format_instructions=FORMAT_INSTRUCTIONS)
    final_prompt = format_instructions.format(tool_names=tool_names, tools=tool_strings)

    agent_kwargs = {
        "system_message": SystemMessage(content=SYSTEM_MESSAGE),
        "extra_prompt_messages": [
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Remember the previously referred table and always try to answer with respect to the previous table."
            ),
            HumanMessagePromptTemplate.from_template(final_prompt),
        ],
    }
    agent = initialize_agent(
        tools, chat_llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=verbose, agent_kwargs=agent_kwargs, memory=memory
    )
    return agent
