from pydantic import BaseModel, Field

from heavyiq.langchain.llms import LLMType


class CallLLMRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to send to LLM")
    temperature: float = Field(0.0, description="Temperature to use for LLM")
    max_tokens: int = Field(256, description="Max tokens to use for LLM response")
    llm_type: LLMType = Field(LLMType.DEFAULT, description="LLM type to use (default, nl_to_sql, sql_to_answer)")
    stop: list[str] | None = Field(default=None, description="An optional list of stop words")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "prompt": "<|prompt|>\nReturn only the name of the capital for the following state, Idaho\n<|answer|>\n",
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "llm_type": "default (or nl_to_sql or sql_to_answer)",
                    "stop": ["is the capital"],
                }
            ]
        }


class CallLLMResponse(BaseModel):
    response: str = Field(..., description="Response from LLM")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "response": "Denver",
                }
            ]
        }
