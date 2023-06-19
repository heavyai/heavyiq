import json
from typing import Optional, Any

from langchain.memory import ConversationBufferMemory
from langchain.chat_models.base import BaseChatModel
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents import AgentOutputParser

from heavynl.config import get_config
from heavynl.langchain import HeavyDB
from heavynl.langchain.agents import HeavyDBToolkit
from heavynl.langchain.llms import get_chat_llm
from heavynl.langchain.agents.messages import HUMAN_MESSAGE, SYSTEM_MESSAGE, FORMAT_INSTRUCTIONS


class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            cleaned_output = text.strip()
            if "```json" in cleaned_output:
                _, cleaned_output = cleaned_output.split("```json")
            if "```" in cleaned_output:
                cleaned_output, _ = cleaned_output.split("```")
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json") :]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```") :]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            cleaned_output = cleaned_output.strip()
            response = json.loads(cleaned_output)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception:
            # bot forgot to speak json, just output the response as a final answer
            return AgentFinish({"output": text.strip()}, text)


def create_conversational_agent(
    heavydb: Optional[HeavyDB] = None, chat_llm: Optional[BaseChatModel] = None, verbose: bool = False
) -> AgentExecutor:
    """This function creates an AgentExecutor instance that uses a language model to generate responses
    based on user inputs. The agent is designed to interact with a HeavyDB instance and has access
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
    tools = toolkit.get_tools()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, ai_prefix="Assistant")
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        human_message=HUMAN_MESSAGE,
        output_parser=CustomOutputParser(),
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=verbose)
