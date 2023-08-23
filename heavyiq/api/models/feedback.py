from typing import Optional

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """
    /submit-feedback endpoint's request schema class.
    """

    feedback_id: str = Field(
        ...,
        description="A unique identifier for a HeavyIQ request that can be used to submit feedback about the response",
    )
    score: float = Field(..., description="From 0.0 (worst) to 1.0 (best)")
    comment: Optional[str] = Field(..., description="An optional comment about the feedback")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "feedback_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "score": 1.0,
                    "comment": "(Optional) This produced a geo-spatial join that was very helpful.",
                }
            ]
        }


class FeedbackResponse(BaseModel):
    """
    /submit-feedback endpoint's response schema class.
    """
