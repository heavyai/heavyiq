from flask import Flask, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue
import openai

from modules.chat_interaction import build_ask_manager, build_answer_manager
import modules.heavydb as db
from modules.logging_utils import default_logger as logger
from modules.sql_utils import extract_sql_from_response, verify_and_attempt_fix_sql

app = Flask(__name__)


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
