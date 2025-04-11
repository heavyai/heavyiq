# contain types w.r.t sql_chain
from typing import TypedDict

from pydantic import BaseModel, Field


class SqlChainInputType(BaseModel):
    """NLtoSQL chain input type."""

    question: str = Field(..., description="Natural Language question.")
    session_id: str = Field(..., description="HeavyDB session ID.")
    tables: list[str] = Field(..., description="List of tables to consider.")

    class Config:
        json_schema_extra = {
            "example": {"question": "What is the revenue?", "session_id": "12345", "tables": ["sales"]}
        }


class SqlChainOutputType(BaseModel):
    """NLtoSQL chain output type."""

    query: str = Field(..., description="Generated query.")
    sql_complexity: int = Field(..., description="sql complexity of the generated query.")
    error: str = Field(..., description="description about the error which might occur during query validation.")


class SqlChainWithCOTOutputType(BaseModel):
    """NLtoSQLwithCOT chain output type."""

    query: str = Field(..., description="Generated query.")
    cot: list[str] = Field(..., description="Chain of Thoughts")
    sql_complexity: int = Field(..., description="sql complexity of the generated query.")
    error: str = Field(..., description="description about the error which might occur during query validation.")


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


class SQLwithScore(BaseModel):
    sql: str = Field(..., description="Generated query.")
    score: float = Field(..., description="sql score")


class SqlMultipleChainOutputType(BaseModel):
    """NLtoSQL chain output type."""

    queries: list[SQLwithScore] = Field(..., description="Generated SQL query with score.")
