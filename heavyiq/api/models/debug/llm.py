from heavyiq.langchain.llms import LLMType

from pydantic import BaseModel, Field


class CallLLMRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to send to LLM")
    temperature: float = Field(0.0, description="Temperature to use for LLM")
    max_tokens: int = Field(256, description="Max tokens to use for LLM response")
    llm_type: LLMType = Field(LLMType.ANY, description="LLM type to use (any, nl_to_sql, sql_to_answer)")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "prompt": "What is the meaning of life?",
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "llm_type": "any (or nl_to_sql or sql_to_answer)",
                }
            ]
        }


class CallLLMResponse(BaseModel):
    response: str = Field(..., description="Response from LLM")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "response": "42",
                }
            ]
        }
