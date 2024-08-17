from pydantic import BaseModel, Field

from .query import RequestBase


class QuestionRequest(RequestBase):
    """
    /question endpoint's request schema class.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    tables: list[str] = Field(
        ..., min_items=1, description="Array of table names to limit the scope of the search/response"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "thread_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "How many states begin with the letter A? What are they?",
                    "tables": ["usa_states"],
                }
            ]
        }


class AutoQuestionRequest(RequestBase):
    """
    /auto/question endpoint's request schema class.
    which was exactly same as TableRequest, QueryRequest schema models.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    allowed_tables: list[str] = Field(
        default=[],
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "thread_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
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
    snippet_ids: list[str] = Field(
        default=[], description="List of snippet IDs where the relevant snippets have been used in the prompt"
    )
    thread_id: str | None = Field(default=None, description="Graph thread id.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "answer": "4 states start with the letter A; Alaska, Arizona, Arkansas, and Alabama.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_result": "[(1, 'Alaska'), (1, 'Arizona'), (1, 'Arkansas'), (1, 'Alabama')]",
                    "sql_complexity": 3,
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "snippet_ids": ["sdsdasafdff"],
                    "thread_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                },
                {
                    "answer": "Generated SQL query resultset exceeds the defined maximum result set size.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_complexity": 3,
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "snippet_ids": ["sdsdasafdff"],
                    "thread_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                },
            ]
        }


class AutoQuestionResponse(QuestionResponse):
    """
    /auto/question endpoint's request schema class.
    """

    tables: list[str] = Field(..., min_items=1, description="List of found table names.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "answer": "4 states start with the letter A; Alaska, Arizona, Arkansas, and Alabama.",
                    "sql": "SELECT COUNT(*) AS num_states, STATE_NAME FROM usa_states WHERE STATE_NAME LIKE 'A%' GROUP BY STATE_NAME;",
                    "sql_result": "[(1, 'Alaska'), (1, 'Arizona'), (1, 'Arkansas'), (1, 'Alabama')]",
                    "sql_complexity": 3,
                    "tables": ["usa_states"],
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "thread_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                }
            ]
        }
