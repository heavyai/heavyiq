from typing import Optional

from pydantic import BaseModel, Field

from heavyiq.langchain.chains.heavydb import CustomExpressionType, TableJoinInfo


class CustomExpressionRequest(BaseModel):
    """
    Custom Expression Request schema.
    """

    question: str = Field(..., description="Natural language question (prompts accepted)")
    type: CustomExpressionType = Field(..., description="The type of the custom expression (measure or dimension)")
    table_name: Optional[str] = Field(None, description="The table name if not a join")
    join_info: Optional[TableJoinInfo] = Field(None, description="Object representing the join information")
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")


class CustomExpressionResponse(BaseModel):
    """
    Custom Expression Response schema.
    """

    expression: str = Field(..., description="The custom expression")
    feedback_id: str = Field(
        ..., description="A unique identifier for this request that can be used to submit feedback about the response"
    )
