from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    /query endpoint's request schema class.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    tables: list[str] = Field(
        ..., min_items=1, description="Array of table names to limit the scope of the search/response"
    )
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "How many states begin with the letter A? What are they?",
                    "tables": ["usa_states"],
                }
            ]
        }


class QueryResponse(BaseModel):
    """
    /query endpoint's response schema class.
    """

    sql: str = Field(..., description="A SQL statement")
    sql_complexity: int = Field(..., description="The complexity level of the SQL statement")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_complexity": 3,
                }
            ]
        }
