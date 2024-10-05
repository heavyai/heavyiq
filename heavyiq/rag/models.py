from datetime import datetime

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


class AddSnippetRequest(BaseModelWithSessionID):
    """
    Add snippet relevant to a particular database.
    """

    snippet: str = Field(..., description="Snippet")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "snippet": "Profitability of a well is defined by sum of oil * current oil price + sum of gas * current gas price - expense column.",
                }
            ]
        }


class UpdateSnippetRequest(BaseModelWithSessionID):
    """
    Update existing snippet.
    """

    snippet_id: str = Field(..., description="Snippet database record ID")
    snippet: str = Field(..., description="Snippet description")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "snippet_id": "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                    "snippet": "Profitability of a well is defined by sum of oil * current oil price + sum of gas * current gas price - expense column.",
                }
            ]
        }


class ListSnippetsRequest(BaseModelWithSessionID):
    """
    Get snippets relevant to a particular database.
    """

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}


class GetSnippetRequest(BaseModelWithSessionID):
    """
    Get relevant snippet by snippet id.
    """

    snippet_id: str = Field(..., description="Snippet database record ID")

    class Config:
        json_schema_extra = {
            "examples": [
                {"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "snippet_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxv"}
            ]
        }


class DeleteSnippetsRequest(BaseModelWithSessionID):
    """
    Delete a list of snippet ids.
    """

    snippet_ids: list[str] = Field(..., description="Fact database record ID", min_length=1)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "snippet_ids": ["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxv", "yyyyyyyyyyyyyyyyyyy"],
                }
            ]
        }


DeleteAllSnippetsRequest = ListSnippetsRequest


class AddSnippetResponse(BaseModel):
    """
    Add Fact response.
    """

    snippet_id: str = Field(..., description="Snippet database record ID")


class Snippet(BaseModel):
    """
    Defines a snippet.
    """

    snippet_id: str = Field(..., description="snippet_id")
    snippet: str = Field(..., description="Snippet")
    created_at: datetime | None = Field(default=None, description="The date and time when the snippet was created")
    updated_at: datetime | None = Field(
        default=None, description="The date and time when the snippet was last modified"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "snippet_id": "9f4ece37daa04a1e956ae084d0ac8088",
                    "snippet": "When asked for the play of a well, use play_designation.",
                    "created_at": "2024-06-19T10:19:05",
                    "updated_at": "2024-06-19T10:32:31",
                }
            ]
        }


SnippetResponse = Snippet


class DeleteSnippetResponse(BaseModel):
    """
    Delete snippets response.
    """

    deleted: bool = Field(default=False, description="Shows whether the snippet gets deleted or not.")


class ListSnippetsResponse(BaseModel):
    """
    Fact list response.
    """

    snippets: list[Snippet] = Field(default=[], description="Snippet list")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "snippets": [
                        {
                            "snippet_id": "9f4ece37daa04a1e956ae084d0ac8088",
                            "snippet": "When asked for the play of a well, use play_designation.",
                            "created_at": "2024-06-19T10:19:05",
                            "updated_at": "2024-06-19T10:32:31",
                        }
                    ]
                }
            ]
        }


class AskSnippetsRequest(BaseModelWithSessionID):
    question: str = Field(..., description="Natural language question")
    only_retrieve: bool = Field(default=True, description="Retrieve Only")
    do_evaluate: bool = Field(default=False, description="Evaluate the response generated by llm")
    use_reranker: bool = Field(
        default=False, description="Set true to enable re-ranker model for sorting resultant nodes."
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "question": "Where should I drill my next well in Colorado to be the most profitable?",
                    "only_retrieve": "true",
                    "use_reranker": "false",
                    "do_evaluate": "false",
                }
            ]
        }


class GetSnippetsfromIndexRequest(BaseModelWithSessionID):
    """
    Get snippets relevant to a particular database.
    """

    limit: int = Field(default=100, description="How many nodes to return..")

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "limit": 100}]}


class OnlySessionIDRequest(BaseModelWithSessionID):
    pass


class SyncNodesRequest(BaseModelWithSessionID):
    """
    Sync nodes with the data from corresponding sources.
    """

    force: bool = Field(default=False, description="Recreate the whole or insert only the misssing data.")


class GetSnippetsfromIndexResponse(BaseModel):
    """
    Get snippets relevant to a particular database.
    """

    class Node(BaseModel):
        content: str = Field(..., description="Node content")
        metadata: dict = Field(..., description="Node metadata")

    nodes: list[Node] = Field(..., description="List of nodes")


ListNodesResponse = GetSnippetsfromIndexResponse

AskSnippetsResponse = AskDocumentResponse


class BulkInsertSnippetsRequest(BaseModelWithSessionID):
    """
    Bulk Insert snippets request.
    """

    snippets: list[str] = Field(..., description="List of snippets.")

    class Config:
        json_schema_extra = {
            "examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "snippets": ["snippet1", "snippet2"]}]
        }


class BulkInsertSnippetsResponse(BaseModel):
    """
    Bulk Insert snippets response.
    """

    snippet_ids: list[str] = Field(default=[], description="List of inserted snippet ids")


class ReEmbedDocumentsRequest(BaseModel):
    """
    Re-embed collection data request.
    """

    collection_name: str = Field(..., description="Name of the collection that you want to re-embed docs on it.")


class ReEmbedDocumentsResponse(BaseModel):
    """
    Re-embed collection data response.
    """

    success: bool = Field(..., description="re-embed docs response")


StatusResponse = ReEmbedDocumentsResponse
