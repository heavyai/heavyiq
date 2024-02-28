from pydantic import BaseModel, Field


class AskHeavyAIDocsRequest(BaseModel):
    question: str = Field(..., description="Question that can be answered using HeavyAI docs")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "question": "Can I configure a HeavyConnect import in the Immerse Data manager?",
                }
            ]
        }


class AskHeavyAIDocsResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question")
    sources: list[str] = Field(..., description="List of sources that were used to generate the answer")
    feedback_id: str = Field(
        ..., description="A unique identifier for this request that can be used to submit feedback about the response"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "answer": "Yes, you can configure a HeavyConnect import in the Immerse Data manager.",
                    "sources": ["https://docs.heavy.ai/immerse/customization"],
                    "feedback_id": "(Optional) xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                }
            ]
        }
