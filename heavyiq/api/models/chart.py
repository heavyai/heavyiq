from typing import Any

from pydantic import BaseModel, Field, Json, root_validator


class GenerateVegaLiteRequest(BaseModel):
    """
    Request model for the /generate-vega endpoint.
    Used to validate user input for generating a Vega-Lite specification.
    """

    question: str = Field(..., description="Generate vega spec based upon the user's question.")
    query: str = Field(..., description="SQL Query which helps to identify the projected columns")
    tables: list[str] = Field(
        ..., min_items=1, description="Array of table names to limit the scope of the search/response"
    )
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxx",
                    "question": "Can you show a bar chart of the average car price by country?",
                    "query": "SELECT c.country AS country_name,AVG(p.price) AS average_price FROM country c JOIN production pr ON c.origin = pr.country JOIN price p ON pr.ID = p.ID GROUP BY c.country ORDER BY average_price DESC;",
                    "tables": ["country", "production", "price"],
                }
            ]
        }


class GenerateVegaLiteResponse(BaseModel):
    """
    Response model for the /generate-vega endpoint.
    """

    vega_lite_spec: Json = Field(..., description="Generated Vega Lite Spec")

    class Config:
        json_schema_extra = {
            "vega_lite_spec": """{ "data": { "name": "table" }, "mark": "bar", "encoding": { "x": { "field": "country_name", "type": "nominal" }, "y": { "field": "average_price", "type": "quantitative" } } }"""
        }


class FixVegaLiteSpecRequest(BaseModel):
    """
    Request model for /fix-vega endpoint.
    """

    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")
    vega_spec: Json = Field(..., description="Vega Lite Spec to fix.")
    errors: list[str] = Field(default=[], description="list of errors which stops the vega to render")
    warnings: list[str] = Field(default=[], description="list of warnings which stops the vega to render")

    @root_validator(pre=True)
    def ensure_errors_or_warnings(cls, values: Any) -> Any:
        if not values.get("errors") and not values.get("warnings"):
            raise ValueError("At least one of 'errors' or 'warnings' must be provided.")
        return values


class FixVegaLiteSpecResponse(BaseModel):
    """
    Response model for /fix-vega endpoint.
    """

    corrected_vega_spec: Json = Field(..., description="Corrected Vega Lite Spec")
