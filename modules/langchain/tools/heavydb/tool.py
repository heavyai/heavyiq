from langchain.tools.base import BaseTool
from pydantic import BaseModel, Extra, Field

from modules.langchain import HeavyDB, heavydb_index
from modules.langchain.templates.tools.heavydb import (
    QueryHeavyDBSchemaToolDescription,
    QueryHeavyDBToolDescription,
    InfoHeavyDBToolDescription,
)


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
    description = QueryHeavyDBSchemaToolDescription

    def _run(self, prompt: str) -> str:
        resp = heavydb_index.query(
            f"Please return the relevant table names (not titles, comma separated) for the following query: {prompt}"
        )
        table_names = resp.strip().split(", ")
        if table_names[0] == "I don't know.":
            return "No information found. Please revise your prompt if you wish to try again."
        try:
            return self.db.get_table_info(table_names)
        except ValueError:
            return "No information found. Please revise your prompt if you wish to try again."

    async def _arun(self, prompt: str) -> str:
        raise NotImplementedError("QueryHeavyDBSchemaTool does not support async")


class QueryHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """Tool for querying a HeavyDB database."""

    name = "query_sql_db"
    description = QueryHeavyDBToolDescription

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
    description = InfoHeavyDBToolDescription

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            return "Error: table names cannot contain single quotes"
        return self.db.get_table_info_no_throw(table_names.strip().split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")
