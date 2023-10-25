import json
import re
from typing import Any, Optional
from collections.abc import Sequence

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.agents import AgentExecutor, AgentOutputParser, BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.conversational_chat.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager, Callbacks
from langchain.chat_models.base import BaseChatModel
from langchain.memory import CombinedMemory, ConversationBufferWindowMemory
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.messages import BaseMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    BaseMessagePromptTemplate,
    BaseChatPromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish, BaseOutputParser
from langchain.tools import BaseTool

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.agents import HeavyDBToolkit
from heavyiq.langchain.callbacks import ConvoAgentCallbackHandler
from heavyiq.langchain.llms import get_chat_llm
from heavyiq.langchain.memory import HeavyIQQueryBufferWindowMemory
from heavyiq.logging_utils import get_heavyiq_logger


SYSTEM_MESSAGE = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."""

HUMAN_MESSAGE = """TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""


CHAT_FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me please, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "thought": string \\ your initial thought
    "action": string \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{{{
    "thought": string \\ your final thoughts
    "action": "Final Answer"
    "action_input": string \\ You should put what you want to return to human
}}}}
```"""


class ChatOutputParser(AgentOutputParser):
    """
    Output Parser specifically used for chat agents.
    """

    def get_format_instructions(self) -> str:
        return CHAT_FORMAT_INSTRUCTIONS

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
            action, action_input, thought = response["action"], response["action_input"], response.get("thought")
            if action == "Final Answer":
                return AgentFinish({"output": action_input, "thought": thought}, text)
            else:
                return AgentAction(action, action_input, cleaned_output)
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
        messages: list[BaseMessagePromptTemplate | BaseChatPromptTemplate | BaseMessage] = [
            SystemMessagePromptTemplate.from_template(system_message),
            # instruct Assistant to pick recently used table for consideration.
            SystemMessagePromptTemplate.from_template(
                """\nIf you're unsure about the table to use, consider referencing any recently used tables from the chat history to help determine the appropriate table for the SQL query.\n\
                    To retrieve the last executed SQL query, you have to pick it from the prompt's SQL history section.
                    User can use a shortcut command `!history` which tells the Agent to pick the last executed SQL query. 
                    """
            ),
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
    An agent executor which stores n success queries on a special HeavyIQQueryBufferWindowMemory memory.
    This memory got retained and will be used upon every run.
    """

    memory: CombinedMemory
    max_iterations: Optional[int] = 10

    @property
    def _heavyiq_buffer_memory(self) -> HeavyIQQueryBufferWindowMemory:
        """
        Gets the instance of heavyiq buffer memory.
        """
        return next(i for i in self.memory.memories if isinstance(i, HeavyIQQueryBufferWindowMemory))

    def _save_to_sql_memory(self, response: dict[str, Any]) -> bool:
        """
        Helps to save the agentexecutor return to HeavyIQQueryBufferWindowMemory.
        """
        heavyiq_buffer_memory = self._heavyiq_buffer_memory
        # save the intermediate success query inside HeavyIQQueryBufferWindowMemory memory
        for agent_action, output in response["intermediate_steps"][::-1]:
            if agent_action.tool == "run_sql_query" and not output.startswith("Error:"):
                inputs = {heavyiq_buffer_memory.input_key: response["input"]}
                outputs = {heavyiq_buffer_memory.output_key: agent_action.tool_input}
                heavyiq_buffer_memory.save_context(inputs, outputs, manual_save=True)
                return True
        return False

    def run(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Custom run method for storing successful query returned from intermediate steps.
        """
        raise NotImplementedError("SaveSuccessQueryAgentExecutor won't support sync run method.")

    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Custom run method for storing successful query returned from intermediate steps asynchronously.
        """
        inputs = {"input": args[0]}
        kwargs = {**kwargs, **inputs}
        response = await self.acall(kwargs, callbacks=callbacks, tags=tags, metadata=metadata)
        status = self._save_to_sql_memory(response=response)
        if status:
            get_heavyiq_logger().debug("Saved the last success query")
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
            1. ConversationBufferWindowMemory used for storing chat history (recent 10 messages, k=10).
            2. HeavyIQQueryBufferWindowMemory used for storing n successful queries.

        Args:
            last_k [int] - last n successfull queries to save to the buffer.
        """
        # pop out any memory being passed
        kwargs.pop("memory", None)
        kwargs.pop("return_intermediate_steps", None)
        chat_history, sql_history = kwargs.pop("chat_history", None), kwargs.pop("sql_history", None)
        convo_memory = ConversationBufferWindowMemory(
            chat_memory=chat_history or ChatMessageHistory(),
            memory_key="chat_history",
            return_messages=True,
            ai_prefix="Assistant",
            input_key="input",
            output_key="output",
            k=10,
        )
        latest_run_sql_memory = HeavyIQQueryBufferWindowMemory(
            chat_memory=sql_history or ChatMessageHistory(),
            memory_key="last_run_sql",
            k=last_k,
            input_key="input",
            return_messages=True,
            output_key="output",
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
        tags=["agent", "conversational_sql_agent"], temperature=0, model=get_config().openai_gpt_model
    )
    heavydb = heavydb or HeavyDB.from_env()
    toolkit = HeavyDBToolkit(db=heavydb)
    tools = toolkit.get_tools()
    return SaveSuccessQueryAgentExecutor.initialize(
        agent_kwargs=dict(
            llm=chat_llm,
            system_message=SYSTEM_MESSAGE,
            human_message=HUMAN_MESSAGE,
            output_parser=ChatOutputParser(),
        ),
        tools=tools,
        last_k=1,
        verbose=verbose,
        callbacks=[ConvoAgentCallbackHandler()],
    )
