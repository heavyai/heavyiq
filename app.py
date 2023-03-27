from flask import Flask, Blueprint, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
import openai

from modules.chat_interaction import build_ask_manager, build_answer_manager
import modules.heavydb as db
from modules.logging_utils import default_logger as logger
from modules.sql_utils import extract_sql_from_response, verify_and_attempt_fix_sql

app = Flask(__name__)
CORS(app)


@app.route("/", methods=("GET", "POST"))
def index() -> ResponseReturnValue:
    if request.method == "POST":
        table_name1 = request.form["table_name1"]
        table_name2 = request.form["table_name2"]
        user_question = request.form["user_question"]

        tables = [table_name1]
        if table_name2 and table_name2 != "":
            tables.append(table_name2)

        logger.info("Request Received")
        logger.info(f"Table(s): {', '.join(tables)}")
        # template options: original, summarized_guidelines, given_instruction, with_examples, added_context
        ask_manager = build_ask_manager(tables, user_question, include_topN=True, template="original")
        logger.info("Built Ask Manager")
        sql_statement = extract_sql_from_response(ask_manager.prompt_ai())
        try:
            sql_statement, sql_result = verify_and_attempt_fix_sql(sql_statement, ask_manager)
            logger.info(f"Valid SQL Parsed: {sql_statement}")
            answer_manager = build_answer_manager(sql_statement, user_question, sql_result)
            answer = answer_manager.prompt_ai()
            return redirect(
                url_for(
                    "index",
                    answer=answer,
                    sql=sql_statement,
                    selected_table_name1=table_name1,
                    selected_table_name2=table_name2,
                    user_question=user_question,
                )
            )
        except openai.InvalidRequestError as error:
            raise error
        except Exception:
            raise Exception("Unable to form valid SQL")

    tables = db.get_tables()
    tables.sort(key=lambda table: table.lower())
    return render_template(
        "index.html",
        answer=request.args.get("answer"),
        sql=request.args.get("sql"),
        tables=tables,
        selected_table_name1=request.args.get("selected_table_name1"),
        selected_table_name2=request.args.get("selected_table_name2"),
        user_question=request.args.get("user_question"),
    )


api_blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
api = Api(
    api_blueprint,
    version="0.1",
    title="HeavyAnalyst API",
    description="A simple REST API for interacting with HeavyNL",
    doc="/docs/",
)
app.register_blueprint(api_blueprint)

table_question_model = api.model(
    "TableQuestion",
    {
        "session_id": fields.String(required=True, description="HeavyDB Session ID", location="json"),
        "tables": fields.List(
            fields.String,
            required=True,
            description="List of table names the question pertains to",
            location="json",
            default=["heavyai_us_counties"],
        ),
        "question": fields.String(
            required=True,
            description="User question",
            location="json",
            default="How many counties start with the letter A?",
        ),
    },
)
# unfortunately the rest package doesn't know how to use models for request parsing so this is a little redundant
# GPT-4 failed to help (gasp)
table_question_parser = reqparse.RequestParser()
table_question_parser.add_argument("session_id", type=str, required=True)
table_question_parser.add_argument("tables", type=str, required=True, action="append")
table_question_parser.add_argument("question", type=str, required=True)


@api.route("/query", doc={"description": "Returns SQL to answer a question about a table(s)"})
class Query(Resource):
    @api.expect(table_question_model)
    def post(self):
        args = table_question_parser.parse_args()
        tables = args["tables"]
        question = args["question"]

        # TODO db_session_id = args["session_id"]
        logger.info("Request Received")
        logger.info(f"Table(s): {', '.join(tables)}")
        ask_manager = build_ask_manager(tables, question, include_topN=True)
        logger.info("Built Ask Manager")
        sql_statement = extract_sql_from_response(ask_manager.prompt_ai())
        try:
            sql_statement, sql_result = verify_and_attempt_fix_sql(sql_statement, ask_manager)
            logger.info(f"Valid SQL Parsed: {sql_statement}")
            answer_manager = build_answer_manager(sql_statement, question, sql_result)
            answer = answer_manager.prompt_ai()
            return {"sql": sql_statement, "answer": answer}
        except openai.InvalidRequestError as error:
            return {"error": str(error)}, 400
        except Exception:
            return {"error": "Unable to find valid SQL"}, 400


@api.route("/question", doc={"description": "Returns NL Answer along with SQL to answer a question about a table(s)"})
class Question(Resource):
    @api.expect(table_question_model)
    def post(self):
        args = table_question_parser.parse_args()
        tables = args["tables"]
        question = args["question"]

        # TODO db_session_id = args["session_id"]
        logger.info("Request Received")
        logger.info(f"Table(s): {', '.join(tables)}")
        # template options: original, summarized_guidelines, given_instruction, with_examples, added_context
        ask_manager = build_ask_manager(tables, question, include_topN=True)
        logger.info("Built Ask Manager")
        sql_statement = extract_sql_from_response(ask_manager.prompt_ai())
        try:
            sql_statement, _ = verify_and_attempt_fix_sql(sql_statement, ask_manager)
            logger.info(f"Valid SQL Parsed: {sql_statement}")
            return {"sql": sql_statement}
        except openai.InvalidRequestError as error:
            return {"error": str(error)}, 400
        except Exception:
            return {"error": "Unable to find valid SQL"}, 400
