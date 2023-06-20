import json
from typing import Optional, Any

from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain.chat_models.base import BaseChatModel
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents import AgentOutputParser

from heavynl.config import get_config
from heavynl.langchain import HeavyDB
from heavynl.langchain.agents import HeavyDBToolkit
from heavynl.langchain.memory import HeavyNLQueryBufferWindowMemory
from heavynl.langchain.callbacks import ConvoAgentCallbackHandler
from heavynl.langchain.llms import get_chat_llm

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

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me please, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": string \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to use here
}}}}
```"""


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
    def create_prompt(cls, *args, **kwargs) -> ChatPromptTemplate:
        # super().create_prompt returns ChatPromptTemplate. Signature mismatch
        chat_prompt: ChatPromptTemplate = super().create_prompt(*args, **kwargs)  # type: ignore
        messages: list = chat_prompt.messages
        messages.insert(2, MessagesPlaceholder(variable_name="last_run_sql"))
        return ChatPromptTemplate(input_variables=chat_prompt.input_variables, messages=chat_prompt.messages)


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
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, ai_prefix="Assistant", input_key="input", output_key="output"
    )
    latest_run_sql_memeory = HeavyNLQueryBufferWindowMemory(
        memory_key="last_run_sql", k=1, input_key="input", return_messages=True, output_key="output"
    )
    memory = CombinedMemory(memories=[memory, latest_run_sql_memeory])

    agent = CustomConversationalChatAgent.from_llm_and_tools(
        llm=chat_llm,
        tools=tools,
        system_message=SYSTEM_MESSAGE,
        human_message=HUMAN_MESSAGE,
        input_variables=["input", "chat_history", "last_run_sql", "agent_scratchpad"],
        output_parser=CustomOutputParser(),
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        return_intermediate_steps=True,
        callbacks=[ConvoAgentCallbackHandler()],
    )


def run_conversational_agent(sql_agent: AgentExecutor, question: str) -> str:
    """
    Function responsible for running the conversational agent.
    """
    heavynl_buffer_memory = next(i for i in sql_agent.memory.memories if isinstance(i, HeavyNLQueryBufferWindowMemory))
    response = sql_agent({"input": question})
    # save the intermediate success query on HeavyNLQueryBufferWindowMemory memory
    for agent_action, output in response["intermediate_steps"][::-1]:
        if agent_action.tool == "run_sql_query" and not output.startswith("Error:"):
            inputs = {heavynl_buffer_memory.input_key: response["input"]}
            outputs = {heavynl_buffer_memory.output_key: agent_action.tool_input}
            heavynl_buffer_memory.save_context(inputs, outputs, manual_save=True)
            break
    return response["output"]
