from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from pydantic import Field

from heavyiq.langchain import HeavyDB
from heavyiq.langchain.agents.tools import QueryHeavyDBTool, RetrieveRelevantSchemasTool, RetrieveTableSchemasTool


class HeavyDBToolkit(BaseToolkit):
    """This toolkit provides a set of tools for querying and retrieving information from a HeavyDB instance.
    It includes tools for retrieving table schemas, finding relevant tables for a query, and executing SQL queries.

    Attributes:
    - db (HeavyDB): A HeavyDB instance to interact with.

    Methods:
    - dialect() -> str: Returns the dialect used by the HeavyDB instance.
    - get_tools() -> list[BaseTool]: Returns a list of tools available in the toolkit."""

    db: HeavyDB = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def dialect(self) -> str:
        """
        Return the dialect used by the HeavyDB instance.

        Returns:
        - str: The dialect used by the HeavyDB instance.
        """
        return self.db.dialect

    def get_tools(self) -> list[BaseTool]:
        """
        Get the tools available in the toolkit.

        Returns:
        - list[BaseTool]: A list of tools for interacting with the HeavyDB instance.
        """
        return [
            RetrieveTableSchemasTool(db=self.db),  # type: ignore
            RetrieveRelevantSchemasTool(db=self.db),  # type: ignore
            QueryHeavyDBTool(db=self.db),  # type: ignore
        ]
