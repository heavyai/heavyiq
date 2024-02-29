from typing import TypedDict

from langchain.pydantic_v1 import BaseModel, Field


class QuestionsChainInputType(BaseModel):
    """Tables to Questions chain input type."""

    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list[str] = Field(
        ...,
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )


class QuestionsChainInputTypedDict(TypedDict):
    session_id: str
    tables: list[str]


class QuestionsChainOutputType(BaseModel):
    """Tables to Questions chain output type"""

    __root__: list[str] = Field(..., description="A list of generated questions.")
