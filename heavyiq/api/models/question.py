from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    /question endpoint's request schema class.
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


class QuestionResponse(BaseModel):
    """
    /question endpoint's response schema class.
    """

    answer: str = Field(..., description="Natural language answer")
    sql: str = Field(..., description="A SQL statement")
    sql_complexity: int = Field(..., description="The complexity level of the SQL statement")
    feedback_id: Optional[str] = Field(
        ..., description="A unique identifier for this request that can be used to submit feedback about the response"
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "answer": "4 states start with the letter A; Alaska, Arizona, Arkansas, and Alabama.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_complexity": 3,
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                }
            ]
        }
