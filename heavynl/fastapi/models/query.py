from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    /query endpoint's request schema class.
    """

    question: str = Field(..., description="Asked NL question.")
    tables: list[str] = Field(..., min_items=1)
    session_id: str = Field(..., max_length=32, min_length=32)

    class Config:
        schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxx",
                    "question": "What is the total population in the USA according to the data in the usa_states table?",
                    "tables": ["usa_states"],
                }
            ]
        }


class QueryResponse(BaseModel):
    """
    /query endpoint's response schema class.
    """

    sql: str
    sql_complexity: int
