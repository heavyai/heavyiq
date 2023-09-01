from pydantic import BaseModel, Field


class DbSessionResponse(BaseModel):
    session_id: str = Field(..., description="HeavyDB Session ID")
