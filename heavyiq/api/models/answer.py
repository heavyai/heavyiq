from langchain.pydantic_v1 import BaseModel, Field


class AnswerRequest(BaseModel):
    """
    /answer endpoint's request schema class.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    query: str = Field(..., description="HeavyDB compatible SQL Query")
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
                    "query": "SELECT COUNT(DISTINCT STATE_NAME) AS Number_of_States, STATE_NAME FROM usa_states\nWHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "tables": ["usa_states"],
                }
            ]
        }


class AnswerResponse(BaseModel):
    """
    /answer endpoint's response schema class.
    """

    answer: str = Field(..., description="Natural language answer")
    sql: str = Field(..., description="A SQL statement")
    sql_result: str = Field(..., description="The result of the SQL statement")
    sql_complexity: int = Field(..., description="The complexity level of the SQL statement")
    feedback_id: str = Field(
        ..., description="A unique identifier for this request that can be used to submit feedback about the response"
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "answer": "4 states start with the letter A; Alaska, Arizona, Arkansas, and Alabama.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_result": "[(1, 'Alaska'), (1, 'Arizona'), (1, 'Arkansas'), (1, 'Alabama')]",
                    "sql_complexity": 3,
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                },
                {
                    "answer": "Generated SQL query resultset exceeds the defined maximum result set size.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_complexity": 3,
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                },
            ]
        }
