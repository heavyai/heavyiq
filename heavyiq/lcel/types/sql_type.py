# contain types w.r.t sql_chain
from langchain.pydantic_v1 import BaseModel, Field


class SqlChainInputType(BaseModel):
    """NLtoSQL chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list = Field(..., description="List of tables to consider.")


class SqlChainOutputType(BaseModel):
    """NLtoSQL chain output type."""

    query: str = Field(..., description="Generated query.")


class SqlChainWithComplexityOutputType(BaseModel):
    """NLtoSQL chain output type with complexity."""

    query: str = Field(..., description="generated query.")
    sql_complexity: int = Field(..., description="sql complexity of the generated query.")
