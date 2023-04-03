from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra, Field

from modules.langchain import HeavyDB, heavydb_index


class BaseHeavyDBTool(BaseModel):
    """Base tool for interacting with a HeavyDB database."""

    db: HeavyDB = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class QueryHeavyDBSchemaTool(BaseHeavyDBTool, BaseTool):
    """Tool for quering an index for a HeavyDB database."""

    name = "query_sql_db_index"
    description = """
    Use this tool to retrieve the schemas for a query.

    The input is a prompt that will be handled by a LLM with access to a HeavyDB index that contains schemas, sample rows, and summaries of each table in the database.
    Prepare a prompt that the LLM can use to fetch the information you require to complete your objective.
    The output will be schemas for the tables that the LLM thinks are relevant to the prompt.
    """

    def _run(self, prompt: str) -> str:
        resp = heavydb_index.query(
            f"Please return the relevant table names (comma separated) for the following query: {prompt}"
        )
        table_names = resp.strip().split(", ")
        if table_names == "I don't know.":
            return "No information found. Please revise your prompt if you wish to try again."
        return self.db.get_table_info(table_names)

    async def _arun(self, prompt: str) -> str:
        raise NotImplementedError("QueryHeavyDBSchemaTool does not support async")


class QueryHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for querying a HeavyDB database."""

    name = "query_sql_db"
    description = """
    ONLY USE WITH KNOWN TABLE SCHEMAS. This tool is for querying HeavyDB databases.

    Input an accurate SQL query for output. Apply query limits. If an error occurs, revise and retry. Be familiar with table schemas and avoid common mistakes, such as:

    - NULL values with NOT IN
    - Using UNION instead of UNION ALL
    - BETWEEN for exclusive ranges
    - Data type mismatch
    - Incorrectly quoting identifiers
    - Wrong number of arguments in functions
    - Incorrect data type casting
    - Improper join columns
    - Reserved SQL keywords as aliases
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
    Provide a comma-separated list of tables as input, and this tool will output the schema and sample rows for those tables.

    Example Input: table1, table2, table3
    Refrain from including single quotes in the table names.
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            return "Error: table names cannot contain single quotes"
        return self.db.get_table_info_no_throw(table_names.strip().split(", "))

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
        return self.db.get_usable_table_names()

    async def _arun(self) -> str:
        raise NotImplementedError("ListSqlDbTool does not support async")
