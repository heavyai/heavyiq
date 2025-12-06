from typing import TypedDict

from pydantic import BaseModel, Field


class QuestionsChainInputType(BaseModel):
    """Tables to Questions chain input type."""

    session_id: str = Field(..., description="HeavyDB session id.")
    tables: list[str] = Field(
        ...,
        description="List of tables to consider.",
    )


class QuestionsChainInputTypedDict(TypedDict):
    session_id: str
    tables: list[str]


class QuestionsChainOutputType(BaseModel):
    """Tables to Questions chain output type"""

    __root__: list[str] = Field(..., description="List of generated questions.")
