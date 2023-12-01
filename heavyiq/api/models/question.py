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


class AutoQuestionRequest(BaseModel):
    """
    /auto/question endpoint's request schema class.
    which was exactly same as TableRequest, QueryRequest schema models.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")
    allowed_tables: list[str] = Field(
        default=[],
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "How many states begin with the letter A? What are they?",
                    "allowed_tables": ["usa_states", "countries"],
                }
            ]
        }


class QuestionResponse(BaseModel):
    """
    /question endpoint's response schema class.
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


class AutoQuestionResponse(QuestionResponse):
    """
    /auto/question endpoint's request schema class.
    """

    ...
