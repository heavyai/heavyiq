from pydantic import BaseModel, Field


class TablesRequest(BaseModel):
    """
    /tables endpoint's request schema class.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "How many states begin with the letter A? What are they?",
                }
            ]
        }


class TablesResponse(BaseModel):
    """
    /tables endpoint's response schema class.
    """

    tables: dict[str, int] = Field(..., description="Dict of tables having table name as key and int (1, 0) as values")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "tables": {"usa_states": 1},
                }
            ]
        }
