from pydantic import BaseModel, Field


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

    answer: str | None = Field(
        default=None, description="Answer returned by LLM by using the retrieved doc sources as context."
    )
    metadata: dict | None = Field(default=None, description="Metadata about the retrieved document sources")
    sources: list[str | tuple[str, dict, float | None]] = Field(
        default=[], description="Optional list of content sources used to derive the answer"
    )
    score: float | None = Field(default=None, description="Answer Evaluation Score")
    passing: bool | None = Field(default=None, description="Evaluation gets passed?")


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

    documents: list[str] = Field(..., description="documents list")

    class Config:
        json_schema_extra = {"examples": [{"documents": ["foo.pdf", "bar.pdf"]}]}


class TableNamesRequest(BaseModelWithSessionID):
    """
    Table names request schema.
    """

    question: str = Field(..., description="Natural language question")

    class Config:
        json_schema_extra = {
            "examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "question": "What is NLP?"}]
        }


class AddFactRequest(BaseModelWithSessionID):
    """
    Add fact relevant to a particular database.
    """

    fact: str = Field(..., description="Fact")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "fact": "Profitability of a well is defined by sum of oil * current oil price + sum of gas * current gas price - expense column.",
                }
            ]
        }


class UpdateFactRequest(BaseModelWithSessionID):
    """
    Update existing fact.
    """

    fact_id: str = Field(..., description="Fact database record ID")
    fact: str = Field(..., description="Fact description")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "fact_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "fact": "Profitability of a well is defined by sum of oil * current oil price + sum of gas * current gas price - expense column.",
                }
            ]
        }


class ListFactsRequest(BaseModelWithSessionID):
    """
    Get facts relevant to a particular database.
    """

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}


class GetFactRequest(BaseModelWithSessionID):
    """
    Get facts relevant to a particular database.
    """

    fact_id: str = Field(..., description="Fact database record ID")

    class Config:
        json_schema_extra = {
            "examples": [
                {"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "fact_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxv"}
            ]
        }


DeleteFactRequest = GetFactRequest
DeleteAllFactsRequest = ListFactsRequest


class AddFactResponse(BaseModel):
    """
    Add Fact response.
    """

    fact_id: str = Field(..., description="Fact database record ID")


class FactResponse(BaseModel):
    """
    Fact response.
    """

    fact: str | None = Field(default=None, description="facts")

    class Config:
        json_schema_extra = {"examples": [{"fact": "asddsssddsdd"}]}


class DeleteFactResponse(BaseModel):
    """
    Delete facts response.
    """

    deleted: bool = Field(default=False, description="Shows whether the object gets deleted or not.")


class ListFactsResponse(BaseModel):
    """
    Fact list response.
    """

    class Fact(BaseModel):
        id: str = Field(..., description="fact_id")
        fact: str = Field(..., description="Fact")

    facts: list[Fact] = Field(default=[], description="Facts list")


class AskFactsRequest(BaseModelWithSessionID):
    question: str = Field(..., description="Natural language question")
    only_retrieve: bool = Field(default=True, description="Retrieve Only")
    do_evaluate: bool = Field(default=False, description="Evaluate the response generated by llm")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "Where should I drill my next well in Colorado to be the most profitable?",
                    "only_retrieve": "true",
                    "do_evaluate": "false",
                }
            ]
        }


class GetFactsfromIndexRequest(BaseModelWithSessionID):
    """
    Get facts relevant to a particular database.
    """

    limit: int = Field(default=100, description="How many nodes to return..")

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "limit": 100}]}


class GetFactsfromIndexResponse(BaseModel):
    """
    Get facts relevant to a particular database.
    """

    class Node(BaseModel):
        content: str = Field(..., description="Node content")
        metadata: dict = Field(..., description="Node metadata")

    nodes: list[Node] = Field(..., description="List of nodes")

    class Config:
        json_schema_extra = {"examples": {"nodes": [{"content": "This is a node's content", "metadata": ""}]}}


AskFactsResponse = AskDocumentResponse
