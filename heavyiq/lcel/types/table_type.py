from pydantic import BaseModel, Field


class TableChainInputType(BaseModel):
    """NLtoTable chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session id.")
    allowed_tables: list[str] = Field(
        default=[],
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )


class TableChainOutputType(BaseModel):
    """NLtoTable chain output type"""

    __root__: dict[str, int] = Field(..., description="A dictionary with table names and presence as integer values.")
