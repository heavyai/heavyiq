from fastapi import UploadFile
from pydantic import BaseModel, Field

from heavyiq.rag.document.database import Document


class BaseModelWithSessionID(BaseModel):
    """
    Pydantic Baseclass with session id field.
    """

    session_id: str = Field(
        ..., description="Search on the documents index where this heavydb sessionid is associated with."
    )


class AskDocumentRequest(BaseModelWithSessionID):
    """
    Ask question about the uploaded documents.
    """

    question: str = Field(..., description="Natural language question")
    filename: str | None = Field(
        default=None,
        description="Provide the filename if you want the question answered within the context of particular file",
    )

    class Config:
        json_schema_extra = {
            "examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "question": "What is NLP?"}]
        }


class AskDocumentResponse(BaseModel):
    """
    AskDocument endpoint's response model.
    """

    answer: str = Field(..., description="Answer returned by LLM by using the retrieved doc sources as context.")
    metadata: dict = Field(..., description="Metadata about the retrieved document sources")
    sources: list[str] = Field(default=[], description="Optional list of content sources used to derive the answer")


class ListFilesRequest(BaseModelWithSessionID):
    """
    List all the uploaded documents specific to the database where the passsed session id belongs to.
    """

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}


class ListFilesResponse(BaseModel):
    """
    List of Documents.
    """

    documents: list[dict] = Field(..., description="documents list")
