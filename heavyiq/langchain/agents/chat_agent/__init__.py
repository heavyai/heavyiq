import re
from typing import Optional

from langchain.agents import AgentExecutor, AgentOutputParser, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, BaseMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.agents import HeavyDBToolkit
from heavyiq.langchain.llms import get_chat_llm

AGENT_PROMPT_TEMPLATE = """You are an agent designed to interact with a SQL database.
Before executing any queries, please consider if the question can be answered by the data in the database.

Available tables: {tables}

Given an input question, you may need to create one or more syntactically correct ANSI SQL queries to run.
You can order the results by a relevant column to return the most interesting examples in the database.
NEVER query for all the columns from a specific table, ONLY ask for the few relevant columns given the question.
If the user doesn't specify a specific number of examples they wish to obtain, please limit your query to at most {top_k} results.
You have access to tools for interacting with the database. Only use the below tools, and only use the information returned by the below tools to construct your final answer.
After using each tool, ask yourself if you have enough information to answer the question and don't use any more tools if you don't need to.
You may need to execute more than one SQL query to arrive at a final conclusion.
If you get an error while executing a query, rewrite the query and try again.

{tools}

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
Always use UNION ALL instead of UNION. If the question does not seem related to the database, just return "I cannot find that information in HeavyDB" as the answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
"""


# Set up a prompt template
class ChatAgentPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: list[BaseTool]
    # The list of tables available to the bot
    tables: list[str]
    # The limit as to how many rows the agent should return
    top_k: int = 10

    def format_messages(self, **kwargs) -> list[BaseMessage]:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tables"] = ", ".join(self.tables)
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["top_k"] = self.top_k
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Check if agent should finish
        if "final answer:" in llm_output.lower():
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.lower().split("final answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def create_sql_agent(
    heavydb: Optional[HeavyDB] = None, top_k: int = 10, chat_llm: Optional[BaseChatModel] = None
) -> AgentExecutor:
    """This function creates an AgentExecutor instance that uses a language model to generate SQL queries
    based on input questions. The agent is designed to interact with a HeavyDB instance and has access
    to a set of tools for querying the database. The agent follows a specific prompt template and
    returns the final answer after executing the necessary SQL queries.

    Parameters:
    - heavydb (Optional[HeavyDB]): A HeavyDB instance to interact with. If not provided, a default
    HeavyDB instance will be created using environment variables.
    - top_k (int, optional): The maximum number of rows the agent should return in its queries.
    Defaults to 10.
    - chat_llm (Optional[BaseChatModel], optional): A language model instance to use for generating
    SQL queries. If not provided, a default ChatOpenAI instance with GPT-4 will be used.

    Returns:
    - AgentExecutor: An AgentExecutor instance configured with the SQL agent and the set of tools
    for interacting with the database."""
    chat_llm = chat_llm or get_chat_llm(
        tags=["agent", "one_time_sql_agent"], temperature=0, model=get_config().openai_gpt_model
    )
    heavydb = heavydb or HeavyDB.from_env()
    toolkit = HeavyDBToolkit(db=heavydb)
    tools = toolkit.get_tools()
    prompt = ChatAgentPromptTemplate(
        template=AGENT_PROMPT_TEMPLATE,
        tools=tools,
        top_k=top_k,  # This omits the `agent_scratchpad`, `tools`, `tool_names`, and `top_k` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        tables=list(heavydb.get_usable_table_names()),
        input_variables=["input", "intermediate_steps"],
    )
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=chat_llm, prompt=prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
