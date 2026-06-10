# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class TablesRequest(BaseModel):
    """
    /tables endpoint's request schema class.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")
    allowed_tables: list[str] = Field(
        default=[],
        description="Optional field representing a list of tables to consider. If this list is empty, all the database tables will be included.",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "How many states begin with the letter A? What are they?",
                    "allowed_tables": ["(Optional) usa_states", "countries"],
                }
            ]
        }


class TablesResponse(BaseModel):
    """
    /tables endpoint's response schema class.
    """

    tables: dict[str, int] = Field(..., description="Dict of tables having table name as key and int (1, 0) as values")
    feedback_id: str = Field(
        default="",
        description="A unique identifier for this request that can be used to submit feedback about the response",
    )
    snippet_ids: list[str] = Field(
        default=[], description="List of snippet IDs where the relevant snippets have been used in the prompt"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "tables": {"usa_states": 1},
                    "snippet_ids": ["saddsasdw12123fd23e"],
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                }
            ]
        }


class TablesToQuestionsRequest(BaseModel):
    """
    /tables-to-questions endpoint's request schema class.
    """

    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")
    tables: list[str] = Field(
        ...,
        description="List of tables to consider.",
    )
    n: int = Field(default=5, ge=1, le=10, description="Number of questions to be generated")
    temperature: float = Field(default=0.5, ge=0.0, le=1.0, description="Number of questions to be generated")
    max_tokens: int = Field(
        default=512, ge=1, le=1024, description="Maxium number of tokens that a LLM can generate (ie. output tokens)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "tables": ["flights_2008"],
                    "n": 5,
                    "temperature": 0.5,
                    "max_tokens": 512,
                }
            ]
        }


class TablesToQuestionsResponse(BaseModel):
    """
    /tables-to-questions endpoint's response schema class.
    """

    questions: list[str] = Field(..., description="List of generated questions.")
    feedback_id: str = Field(
        default="",
        description="A unique identifier for this request that can be used to submit feedback about the response",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "questions": ["What are the top 5 airlines by number of flights operated?"],
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                }
            ]
        }
