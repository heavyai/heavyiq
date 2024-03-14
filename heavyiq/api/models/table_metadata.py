from pydantic import BaseModel, Field


class GenerateTableMetadataRequest(BaseModel):
    """
    /generate-table-metadata endpoint's request schema class.
    """

    table_name: str = Field(..., description="Name of the table for which metadata needs to be generated")
    session_id: str = Field(..., max_length=32, min_length=32, description="Valid HeavyDB Session ID")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    "table_name": "usa_states",
                }
            ]
        }


class GenerateTableMetadataResponse(BaseModel):
    """
    /generate-table-metadata endpoint's response schema class.
    """

    table_name: str = Field(..., description="Name of the table for which metadata gets generated")
    summary: str = Field(..., description="description of the table as a whole")
    columns: dict[str, str] = Field(..., description="Description of each column.")
    feedback_id: str = Field(
        ..., description="A unique identifier for this request that can be used to submit feedback about the response"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "table_name": "us_pois_safegraph",
                    "summary": 'The "us_pois_safegraph" table is designed to store information about points of interest (POIs) in the United States.',
                    "columns": {
                        "safegraph_place_id": "A unique identifier for each place in the dataset.",
                        "parent_safegraph_place_id": "The safegraph_place_id of the parent place, if applicable.",
                        "safegraph_brand_ids": "A comma-separated list of brand IDs associated with the place.",
                        "location_name": "The name of the location.",
                        "brands": "The name of the brand associated with the location.",
                        "top_category": "The top-level category of the location.",
                        "sub_category": "The sub-category of the location.",
                        "naics_code": "The NAICS code associated with the location.",
                        "latitude": "The latitude coordinate of the location.",
                        "longitude": "The longitude coordinate of the location.",
                        "street_address": "The street address of the location.",
                        "primary_number": "The primary number of the street address.",
                        "street_predirection": "The pre-directional of the street address.",
                        "street_name": "The name of the street.",
                        "street_postdirection": "The post-directional of the street address.",
                        "street_suffix": "The suffix of the street address.",
                        "city": "The city where the location is located.",
                        "region": "The region (state) where the location is located.",
                        "postal_code": "The postal code of the location.",
                        "open_hours": "The opening hours of the location in JSON format.",
                        "category_tags": "Tags associated with the location's category.",
                        "polygon_wkt": "The polygon representation of the location in Well-Known Text (WKT) format.",
                        "polygon_class": "The class of the polygon (OWNED_POLYGON or SHARED_POLYGON).",
                        "phone_number": "The phone number of the location.",
                        "is_synthetic": "Indicates if the location is synthetic (false or true).",
                        "includes_parking_lot": "Indicates if the location includes a parking lot (false or true).",
                        "iso_country_code": "The ISO country code of the location.",
                    },
                }
            ]
        }
