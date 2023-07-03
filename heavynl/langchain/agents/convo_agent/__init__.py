import json
import re
from typing import Any, Optional
from collections.abc import Sequence

from langchain.agents import AgentExecutor, AgentOutputParser, BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.conversational_chat.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chat_models.base import BaseChatModel
from langchain.memory import CombinedMemory, ConversationBufferMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, BaseOutputParser
from langchain.tools import BaseTool

from heavynl.config import get_config
from heavynl.langchain import HeavyDB
from heavynl.langchain.agents import HeavyDBToolkit
from heavynl.langchain.callbacks import ConvoAgentCallbackHandler
from heavynl.langchain.llms import get_chat_llm
from heavynl.langchain.agents.messages import HUMAN_MESSAGE, SYSTEM_MESSAGE, FORMAT_INSTRUCTIONS
from heavynl.langchain.memory.query_memory import HeavyNLQueryBufferWindowMemory
from heavynl.logging_utils import get_heavynl_logger


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


class CustomConversationalChatAgent(ConversationalChatAgent):
    """
    Subclass of ConversationalChatAgent to include last_run_sql message inside ChatPromptTemplate messages which
    again passed as input to llm.
    """

    @classmethod
    def create_prompt(
        cls: type["CustomConversationalChatAgent"],
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[list[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join([f"> {tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(format_instructions=_output_parser.get_format_instructions())
        final_prompt = format_instructions.format(tool_names=tool_names, tools=tool_strings)
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad", "last_run_sql"]
        message_without_input, input_message = re.split(r"\b(?=USER'S INPUT)", final_prompt)
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(message_without_input),
            SystemMessagePromptTemplate.from_template("Chat History:\n------------------\n"),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessagePromptTemplate.from_template("\nSQL History:\n------------------\n"),
            MessagesPlaceholder(variable_name="last_run_sql"),
            HumanMessagePromptTemplate.from_template("\n" + input_message),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)


class SaveSuccessQueryAgentExecutor(AgentExecutor):
    """
    An agent executor which stores n success queries on a special HeavyNLQueryBufferWindowMemory memory.
    This memory got retained and will be used upon every run.
    """

    @property
    def _heavynl_buffer_memory(self) -> HeavyNLQueryBufferWindowMemory:
        """
        Gets the instance of heavynl buffer memory.
        """
        return next(i for i in self.memory.memories if isinstance(i, HeavyNLQueryBufferWindowMemory))

    def _save_to_sql_memory(self, response: dict[str, Any]) -> bool:
        """
        Helps to save the agentexecutor return to HeavyNLQueryBufferWindowMemory.
        """
        heavynl_buffer_memory = self._heavynl_buffer_memory
        # save the intermediate success query inside HeavyNLQueryBufferWindowMemory memory
        for agent_action, output in response["intermediate_steps"][::-1]:
            if agent_action.tool == "run_sql_query" and not output.startswith("Error:"):
                inputs = {heavynl_buffer_memory.input_key: response["input"]}
                outputs = {heavynl_buffer_memory.output_key: agent_action.tool_input}
                heavynl_buffer_memory.save_context(inputs, outputs, manual_save=True)
                return True
        return False

    def run(self, question: str, callbacks: Callbacks = None, tags: list[str] | None = None, **kwargs: Any) -> str:
        """
        Custom run method for storing successful query returned from intermediate steps.
        """
        inputs = {"input": question}
        response = self.__call__(inputs=inputs, callbacks=callbacks, tags=tags, **kwargs)
        status = self._save_to_sql_memory(response=response)
        if status:
            get_heavynl_logger().debug("Saved the last success query")
        return response["output"]

    @classmethod
    def from_agent_and_tools(
        cls: type["SaveSuccessQueryAgentExecutor"],
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        last_k: int = 1,
        **kwargs: Any,
    ) -> AgentExecutor:
        """
        Create from agent and tools.
        Any memory or return_intermediate_steps passed or befing popped out and the
        relevant default values has to be used. Why because, we need the executor to
        return intermediate_steps for catching the right query and also the memory should
        be an instance of CombinedMemory
            1. ConversationBufferMemory used for storing chat history.
            2. HeavyNLQueryBufferWindowMemory used for storing n successful queries.

        Args:
            last_k [int] - last n successfull queries to save to the buffer.
        """
        # pop out any memory being passed
        kwargs.pop("memory", None)
        kwargs.pop("return_intermediate_steps", None)
        convo_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            ai_prefix="Assistant",
            input_key="input",
            output_key="output",
        )
        latest_run_sql_memory = HeavyNLQueryBufferWindowMemory(
            memory_key="last_run_sql", k=last_k, input_key="input", return_messages=True, output_key="output"
        )
        memory = CombinedMemory(memories=[convo_memory, latest_run_sql_memory])
        return cls(
            agent=agent,
            tools=tools,
            callback_manager=callback_manager,
            memory=memory,
            return_intermediate_steps=True,
            **kwargs,
        )

    @classmethod
    def initialize(cls: type["SaveSuccessQueryAgentExecutor"], agent_kwargs: dict[str, Any], **kwargs) -> AgentExecutor:
        """
        Initializes SaveSuccessQueryAgentExecutor class.

        parameters:
            agent_kwargs - Dict of kwargs passed to the agent.
        """
        tools = kwargs.pop("tools") or agent_kwargs.pop("tools")
        agent = CustomConversationalChatAgent.from_llm_and_tools(tools=tools, **agent_kwargs)  # type: ignore
        return SaveSuccessQueryAgentExecutor.from_agent_and_tools(agent=agent, tools=tools, **kwargs)


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
    return SaveSuccessQueryAgentExecutor.initialize(
        agent_kwargs=dict(
            llm=chat_llm,
            system_message=SYSTEM_MESSAGE,
            human_message=HUMAN_MESSAGE,
            output_parser=CustomOutputParser(),
        ),
        tools=tools,
        last_k=1,
        verbose=verbose,
        callbacks=[ConvoAgentCallbackHandler()],
    )
