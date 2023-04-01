from flask_restx import Resource
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.api import api
from modules.api.rest_models import (
    table_question_model,
    query_response_model,
    question_response_model,
    error_response_model,
    table_question_parser,
)
from modules.langchain.heavydb import HeavyDB
from modules.langchain.chains.heavydb.base import NLtoSQLChain, NLtoAnswerChain
from modules.logging_utils import default_logger as logger

MODEL_NAME = "text-davinci-003"


def build_llm(model: str = MODEL_NAME) -> ChatOpenAI | OpenAI:
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        return ChatOpenAI(model_name=model, client=None)
    else:
        return OpenAI(
            model_name=model,
            client=None,
            temperature=0.2,
        )


@api.route("/query", doc={"description": "Returns SQL to answer a question about a table(s)"})
class Query(Resource):
    @api.expect(table_question_model)
    @api.response(200, "Success", query_response_model)
    @api.response(500, "Error", error_response_model)
    def post(self):
        try:
            args = table_question_parser.parse_args()
            tables = args["tables"]
            question = args["question"]
            # db_session_id = args["session_id"]
            logger.info("Request Received")
            logger.info(f"Table(s): {', '.join(tables)}")
            llm = build_llm()
            # db = HeavyDB.from_session(db_session_id, include_tables=tables)
            db = HeavyDB.from_env(include_tables=tables)
            chain = NLtoSQLChain(llm=llm, database=db, verbose=True)
            with get_openai_callback() as cb:
                res = chain(question)
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Tokens: {cb.total_tokens}")
            return {"sql": res[chain.output_key]}, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route("/question", doc={"description": "Returns NL Answer along with SQL to answer a question about a table(s)"})
class Question(Resource):
    @api.expect(table_question_model)
    @api.response(200, "Success", question_response_model)
    @api.response(500, "Error", error_response_model)
    def post(self):
        try:
            args = table_question_parser.parse_args()
            tables = args["tables"]
            question = args["question"]
            # db_session_id = args["session_id"]
            logger.info("Request Received")
            logger.info(f"Table(s): {', '.join(tables)}")
            llm = build_llm()
            # db = HeavyDB.from_session(db_session_id, include_tables=tables)
            db = HeavyDB.from_env(include_tables=tables)
            chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
            with get_openai_callback() as cb:
                res = chain(question)
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Tokens: {cb.total_tokens}")
            return {"answer": res[chain.output_answer_key], "sql": res[chain.output_sql_key]}, 200
        except Exception as e:
            return {"error": str(e)}, 500
