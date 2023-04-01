from flask_restx import fields, reqparse

from modules.api import api

table_question_model = api.model(
    "TableQuestion",
    {
        "session_id": fields.String(required=True, description="HeavyDB Session ID", location="json"),
        "tables": fields.List(
            fields.String,
            required=True,
            description="List of table names the question pertains to",
            location="json",
            default=["usa_states"],
        ),
        "question": fields.String(
            required=True,
            description="User question",
            location="json",
            default="How many states start with the letter A?",
        ),
    },
)
# unfortunately the rest package doesn't know how to use models for request parsing so this is a little redundant
# GPT-4 failed to help (gasp)
table_question_parser = reqparse.RequestParser()
table_question_parser.add_argument("session_id", type=str, required=True)
table_question_parser.add_argument("tables", type=str, required=True, action="append")
table_question_parser.add_argument("question", type=str, required=True)

error_response_model = api.model(
    "ErrorResponse",
    {
        "error": fields.String(description="Error message"),
    },
)

query_response_model = api.model(
    "QueryResponse",
    {
        "sql": fields.String(description="Generated SQL query"),
    },
)

question_response_model = api.model(
    "QuestionResponse",
    {
        "answer": fields.String(description="Natural Language answer"),
        "sql": fields.String(description="Generated SQL query"),
    },
)
