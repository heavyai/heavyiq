# contain types w.r.t sql_chain
from typing import TypedDict

from langchain.pydantic_v1 import BaseModel, Field


class SqlChainInputType(BaseModel):
    """NLtoSQL chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list = Field(..., description="List of tables to consider.")


class SqlChainOutputType(BaseModel):
    """NLtoSQL chain output type."""

    query: str = Field(..., description="Generated query.")
    sql_complexity: int = Field(..., description="sql complexity of the generated query.")


class SqlChainIntermediateType(BaseModel):
    """NLtoSQL chain intermediate type, ie. query retry input type."""

    error: str
    sql_cmd: str
    session_id: str
    question: str
    max_revisions: int
    tables: list[str]


class SqlChainIntermediateDict(TypedDict):
    error: str
    sql_cmd: str
    session_id: str
    question: str
    max_revisions: int
    tables: list[str]
