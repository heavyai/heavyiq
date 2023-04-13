from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from pydantic import Field

from modules.langchain.heavydb import HeavyDB
from modules.langchain.agents.tools import QueryHeavyDBTool, RetrieveRelevantSchemasTool, RetrieveTableSchemasTool


class HeavyDBToolkit(BaseToolkit):
    """Toolkit for interacting with HeavyDB databases."""

    db: HeavyDB = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            RetrieveTableSchemasTool(db=self.db),
            RetrieveRelevantSchemasTool(db=self.db),
            QueryHeavyDBTool(db=self.db),
        ]
