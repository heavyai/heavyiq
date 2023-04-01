"""Toolkit for interacting with a HeavyDB database."""
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from pydantic import Field

from modules.langchain.heavydb import HeavyDB
from modules.langchain.tools.heavydb.tool import (
    InfoHeavyDBTool,
    QueryHeavyDBTool,
    QueryHeavyDBSchemaTool,
)


class HeavyDBToolkit(BaseToolkit):
    """Toolkit for interacting with HeavyDB databases."""

    db: HeavyDB = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QueryHeavyDBTool(db=self.db),
            InfoHeavyDBTool(db=self.db),
            QueryHeavyDBSchemaTool(db=self.db),
        ]
