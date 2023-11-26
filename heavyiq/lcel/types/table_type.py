from langchain.pydantic_v1 import BaseModel, Field


class TableChainInputType(BaseModel):
    """NLtoTable chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session id.")


class TableChainOutputType(BaseModel):
    """NLtoTable chain output type"""

    __root__: dict[str, int] = Field(..., description="A dictionary with table names and presence as integer values.")
