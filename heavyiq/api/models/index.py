from pydantic import BaseModel, Field


class UpdateIndexRequest(BaseModel):
    """
    /update-index endpoint's request schema class.
    """

    session_id: str | None = Field(default=None, description="[Optional] Valid HeavyDB Session ID")

    class Config:
        json_schema_extra = {"examples": [{"session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}, {}]}


class UpdateIndexResponse(BaseModel):
    """
    /update-index endpoint's response schema class.
    """

    status: str = Field(..., description="Status of the updateindex request, (started or in-progress)")
