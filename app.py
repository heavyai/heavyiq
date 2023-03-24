import re

from flask import Flask, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue
from heavyai import Cursor
import openai

from modules.chat_manager import ChatManager
import modules.heavydb as db
from modules.prompt_builder import PromptBuilder

app = Flask(__name__)


MAX_SQL_TRY_COUNT = 3


def build_chatgpt_system_ask_prompt(table_name: str) -> str:
    system_prompt = PromptBuilder()
    system_prompt.add_part(
        "intro", f"A SQL table named {table_name} is stored in the database with the following schema:"
    )
    system_prompt.add_part("schema", db.get_table_schema(table_name))
    system_prompt.add_part(
        "instruction",
        """Translate the following instruction into a SQL query on this table. Moderate preference to return a single row with the answer. Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Only output a SQL query and nothing else.""",
    )
    return system_prompt.build()


# alternate prompt suggested by GPT-4
def build_chatgpt_system_ask_prompt_summarized_guidelines(table_name: str) -> str:
    system_prompt = PromptBuilder()
    system_prompt.add_part(
        "intro", f"A SQL table named {table_name} is stored in the database with the following schema:"
    )
    system_prompt.add_part("schema", db.get_table_schema(table_name))
    system_prompt.add_part(
        "instruction",
        """Translate the following instruction into a SQL query on this table. Return a single row with the answer if possible, exclude null values, and avoid reserved SQL keywords as aliases. Only output a SQL query and nothing else.""",
    )
    return system_prompt.build()


# alternate prompt suggested by GPT-4
def build_chatgpt_system_ask_prompt_added_context(table_name: str) -> str:
    system_prompt = PromptBuilder()
    system_prompt.add_part(
        "intro",
        f"As a language model, your task is to translate natural language instructions into SQL queries using the {table_name} table in the database. The table has the following schema:",
    )
    system_prompt.add_part("schema", db.get_table_schema(table_name))
    system_prompt.add_part(
        "instruction",
        """Given a specific instruction, create an SQL query on this table, with a moderate preference for a single row answer. Exclude null values from the results and avoid using reserved SQL keywords as aliases. Only output a SQL query and nothing else.""",
    )
    return system_prompt.build()


# alternate prompt suggested by GPT-4
def build_chatgpt_system_ask_prompt_with_examples(table_name: str) -> str:
    system_prompt = PromptBuilder()
    system_prompt.add_part(
        "intro", f"A SQL table named {table_name} is stored in the database with the following schema:"
    )
    system_prompt.add_part("schema", db.get_table_schema(table_name))
    system_prompt.add_part(
        "instruction",
        """Translate natural language instructions into SQL queries on this table (e.g., "Find the population of a country given its name."). Return a single row with the answer if possible, exclude null values, and avoid reserved SQL keywords as aliases. Only output a SQL query and nothing else.""",
    )
    return system_prompt.build()


def clean_gpt_sql_statement(stmt: str) -> str:
    stmt = stmt.replace("\n", " ").replace("\r", " ")
    pattern = r"(SELECT.+?;)"
    result = re.search(pattern, stmt, re.IGNORECASE)
    if result:
        return result.group(0)
    raise Exception("No SELECT statement found in provided string")


def verify_and_attempt_fix_sql(
    sql_statement: str, chat_manager: ChatManager, remaining_tries: int = MAX_SQL_TRY_COUNT
) -> tuple[str, Cursor]:
    try:
        return sql_statement, db.conn.execute(sql_statement)
    except Exception as error:
        app.logger.info(f"Invalid SQL: {sql_statement}")
        app.logger.debug(str(error))
        if remaining_tries <= 0:
            raise error
        app.logger.info("Trying again...")
        chat_manager.add_message("user", str(error))
        sql_statement = clean_gpt_sql_statement(chat_manager.prompt_ai())
        return verify_and_attempt_fix_sql(sql_statement, chat_manager, remaining_tries=remaining_tries - 1)


def build_ask_manager(table_name: str, user_question: str) -> ChatManager:
    """Returns a chat manager primed to prompt ai for the SQL query that can be used to answer user_question"""
    chat_manager = ChatManager(logger=app.logger)
    chat_manager.add_message("system", build_chatgpt_system_ask_prompt(table_name))
    chat_manager.add_message("user", user_question)
    return chat_manager


def build_chatgpt_system_answer_prompt(sql_result: Cursor) -> str:
    column_names = [d[0] for d in sql_result.description]
    prompt_builder = PromptBuilder(delimiter="\n")
    results = sql_result.fetchall()
    prompt_builder.add_part("header", ",".join(column_names))
    for row_idx, row in enumerate(results):
        prompt_builder.add_part(f"result-table-row-{row_idx}", ",".join([str(col) for col in row]))

    app.logger.info("==== SQL Results ====")
    for _, part in prompt_builder._prompt_parts:
        app.logger.info(part)
    return prompt_builder.build()


def build_answer_manager(sql_statement: str, user_question: str, sql_result: Cursor) -> ChatManager:
    """Returns a chat manager primed to prompt ai for the answer to user_question using sql_results"""
    chat_manager = ChatManager(logger=app.logger)
    chat_manager.add_message(
        "system",
        f"From the SQL query {sql_statement} in response to the following question: '{user_question}', we received the following result back from the database. Explain this result in plain English. Give a brief answer to the question without explanation of the SQL. Do not explain what the SQL means.",
    )
    chat_manager.add_message("user", build_chatgpt_system_answer_prompt(sql_result))
    return chat_manager


@app.route("/", methods=("GET", "POST"))
def index() -> ResponseReturnValue:
    if request.method == "POST":
        table_name = request.form["table_name"]
        user_question = request.form["user_question"]
        app.logger.info("Request Received")
        ask_manager = build_ask_manager(table_name, user_question)
        app.logger.info("Built Ask Manager")
        sql_statement = clean_gpt_sql_statement(ask_manager.prompt_ai())
        try:
            sql_statement, sql_result = verify_and_attempt_fix_sql(sql_statement, ask_manager)
            app.logger.info(f"Valid SQL Parsed: {sql_statement}")
            answer_manager = build_answer_manager(sql_statement, user_question, sql_result)
            answer = answer_manager.prompt_ai()
            return redirect(url_for("index", answer=answer, selected_table_name=table_name))
        except openai.InvalidRequestError as error:
            raise error
        except Exception:
            raise Exception("Unable to form valid SQL")

    tables = db.get_tables()
    tables.sort(key=lambda table: table.lower())
    return render_template(
        "index.html",
        answer=request.args.get("answer"),
        tables=tables,
        selected_table_name=request.args.get("selected_table_name"),
    )
