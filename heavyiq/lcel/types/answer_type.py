from langchain.pydantic_v1 import BaseModel, Field


class AnswerChainInputType(BaseModel):
    """NLtoAnswer chain input type."""

    question: str = Field(..., description="NL question.")
    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list[str] = Field(..., description="List of table names to consider for SQL generation.")


class AnswerChainOutputType(BaseModel):
    """NLtoAnswer chain output type."""

    sql: str = Field(..., description="Generated SQL query.")
    sql_complexity: int = Field(..., description="SQL complexity of the genarted query.")
    results: str = Field(..., description="Result of the generated query.")
    answer: str = Field(..., description="Output produced by the llm using the HeavyDB result.")
    fail_reason: str = Field(..., description="Contain reason about why the SQLtoAnswer LLM failed to generate answer.")
