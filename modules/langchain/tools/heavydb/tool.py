from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra, Field, validator

from modules.langchain import HeavyDB, heavydb_index
from modules.langchain.tools.heavydb.prompts import QUERY_CHECKER


class BaseHeavyDBTool(BaseModel):
    """Base tool for interacting with a HeavyDB database."""

    db: HeavyDB = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class QueryHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for querying a HeavyDB database."""

    name = "query_sql_db"
    description = """
    You MAY NOT use this tool unless you know the schema of the tables you are querying. Use the schema_sql_db tool to get table schemas.

    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    Don't forget to apply a limit to your query.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    Do not use this tool unless you know the table schemas.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        try:
            self.db.validate_query(query)  # validate the query before running it
            return self.db.run_no_throw(query)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QueryHeavyDBTool does not support async")


class InfoHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for getting metadata about a HeavyDB database."""

    name = "schema_sql_db"
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.

    Example Input: table1, table2, table3
    Do not include single quotes in the table names.
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            return "Error: table names cannot contain single quotes"
        return self.db.get_table_info_no_throw(table_names.split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for listing tables in a HeavyDB database."""

    name = "list_tables_sql_db"
    description = """
    Input to this tool is nothing, output is a list of tables in the database.
    """

    def _run(self, _: str) -> str:
        """Get the list of tables in the database."""
        return self.db.get_table_names()

    async def _arun(self) -> str:
        raise NotImplementedError("ListSqlDbTool does not support async")


class CanQuestionBeAnsweredHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for using an index to check if a question can be answered in a HeavyDB database."""

    name = "can_question_be_answered_sql_db"
    description = """
    Use this tool first to determine if a question can be answered. If it can, you may continue to list_relevant_tables_sql_db.

    Input to this tool is a question.
    Output is if the question can be answered. If the response contains table names, you can skip list_relevant_tables_sql_db.
    If the question cannot be answered, do not run list_relevant_tables_sql_db. You can not answer the question. Do not try. Do not continue.
    """

    def _run(self, question: str) -> str:
        query = f"Can information about the following prompt potentially be found in the database? If so which tables should be checked?: {question}"
        res = heavydb_index.query(query)
        return res

    async def _arun(self) -> str:
        raise NotImplementedError("CanQuestionBeAnsweredHeavyDBTool does not support async")


class ListRelevantTablesHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for using an index to list relevant tables in a HeavyDB database."""

    name = "list_relevant_tables_sql_db"
    description = """
    Did you run can_question_be_answered_sql_db first? If not, do that first.
    Input to this tool is a question.
    Output is if the question can be answered and a list of tables in the database that may contain relevant information.
    If the question cannot be answered, do not continue and state that the question cannot be answered.
    """

    def _run(self, question: str) -> str:
        # retriever = heavydb_index.vectorstore.as_retriever()
        # docs = retriever.get_relevant_documents(question)
        # table_names = ", ".join(list(set([doc.metadata["source"] for doc in docs])))
        # return table_names

        # the following tried to answer the question using sources found outside of HeavyDB
        # question: what is the population of alabama
        # expected source: usa_states
        # provided source: US Census Bureau
        res = heavydb_index.query_with_sources(question)
        table_names = ", ".join(list(set(res["sources"].split(", "))))
        return f"Answer: {res['answer']}\nPotentially Relevant Tables: {table_names}"

    async def _arun(self) -> str:
        raise NotImplementedError("ListRelevantTablesHeavyDBTool does not support async")


# This may need to be adapted to check HeavyDB's specific SQL dialect
class QueryCheckerTool(BaseHeavyDBTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm_chain: LLMChain = Field(
        default_factory=lambda: LLMChain(
            llm=OpenAI(temperature=0),
            prompt=PromptTemplate(template=QUERY_CHECKER, input_variables=["query", "dialect"]),
        )
    )
    name = "query_checker_sql_db"
    description = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @validator("llm_chain")
    def validate_llm_chain_input_variables(cls, llm_chain: LLMChain) -> LLMChain:
        """Make sure the LLM chain has the correct input variables."""
        if llm_chain.prompt.input_variables != ["query", "dialect"]:
            raise ValueError("LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']")
        return llm_chain

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, dialect=self.db.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.db.dialect)
