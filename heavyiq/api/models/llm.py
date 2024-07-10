from pydantic import BaseModel, Field

from heavyiq.langchain.llms import LLMType


class CallLLMRequest(BaseModel):
    question: str = Field(..., description="Actual NL question.")
    temperature: float = Field(0.0, description="Temperature to use for LLM")
    max_tokens: int = Field(768, description="Max tokens to use for LLM response")
    guided_choice: list[str] | None = Field(default=None, description="An optional list of allowed outputs")
    guided_regex: str | None = Field(default=None, description="An optional regex pattern to guide output")
    stop: list[str] | None = Field(default=None, description="An optional list of stop words")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "question": "Return whether the following is a city, state, or country: Idaho",
                    "temperature": 0.0,
                    "max_tokens": 768,
                    "guided_choice": ["city", "state", "country"],
                    "stop": ["is the capital"],
                }
            ]
        }


class CallLLMResponse(BaseModel):
    response: str = Field(..., description="Response from LLM")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "response": "Denver",
                }
            ]
        }
