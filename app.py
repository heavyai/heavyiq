from functools import lru_cache
import os
from typing import NamedTuple, Literal

import openai
from flask import Flask, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue

from heavyai import Connection, Cursor, connect
from heavydb._parsers import ColumnDetails

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_TOP_K = 10
MAX_SQL_TRY_COUNT = 3
CHAT_MODEL = "gpt-3.5-turbo"

heavydb_host = os.getenv("HEAVYDB_HOST")
heavydb_port = os.getenv("HEAVYDB_PORT")
heavydb_dbname = os.getenv("HEAVYDB_DBNAME")
heavydb_username = os.getenv("HEAVYDB_USERNAME")
heavydb_password = os.getenv("HEAVYDB_PASSWORD")
heavydb_protocol = os.getenv("HEAVYDB_PROTOCOL")

conn: Connection = connect(
    user=heavydb_username, password=heavydb_password, host=heavydb_host, port=heavydb_port, dbname=heavydb_dbname
)


PromptPart = NamedTuple("PromptPart", [("label", str), ("content", str)])


class PromptBuilder:
    """Simple prompt builder that allows prompts to be pieced together"""

    _prompt_parts: list[PromptPart]

    def __init__(self, delimiter: str = "\n\n"):
        self._prompt_parts = []
        self.delimiter = delimiter

    def add_part(self, label: str, content: str):
        self._prompt_parts.append(PromptPart(label, content))

    def list_parts(self) -> list[str]:
        return [part.label for part in self._prompt_parts]

    def build(self) -> str:
        return self.delimiter.join([part.content for part in self._prompt_parts])


ChatMessageRole = Literal["system", "assistant", "user"]
ChatMessage = NamedTuple("ChatMessage", [("role", ChatMessageRole), ("content", str)])


class ChatManager:
    _messages: list[ChatMessage]

    def __init__(self, model: str = CHAT_MODEL):
        self.model = model
        self._messages = []

    def add_message(self, role: ChatMessageRole, content: str):
        self._messages.append(ChatMessage(role, content))

    def prompt_ai(self) -> str:
        app.logger.info("==== Prompting OpenAI ====")
        for message in self._messages:
            app.logger.debug(f"[{message.role}] {message.content}")
        response = openai.ChatCompletion.create(model=self.model, messages=self.messages)
        app.logger.info("==== OpenAI Response Received ====")
        content = response["choices"][0]["message"]["content"]
        app.logger.debug(f"Response: {content}")
        self.add_message("assistant", content)
        return content

    @property
    def messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]


@lru_cache
def get_tables() -> list[str]:
    return conn.get_tables()


@lru_cache
def get_table_details(table_name: str) -> list[ColumnDetails]:
    return conn.get_table_details(table_name)


def get_table_schema(table_name: str) -> str:
    table_details: list[ColumnDetails] = get_table_details(table_name)
    res = f"CREATE TABLE {table_name} (\n"
    res += ",\n".join([f"{col.name} {col.type}" for col in table_details]) + "\n"
    res += ");\n"
    return res


def get_table_string_columns(table_name: str) -> list[str]:
    table_details = get_table_details(table_name)
    return [col.name for col in table_details if col.type == "STR" and col.encoding == "DICT"]


@lru_cache
def get_top_k_vals(table_name: str, column_name: str, top_k: int = DEFAULT_TOP_K) -> list[str | bool | int | float]:
    sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY {column_name} ORDER BY COUNT(*) DESC LIMIT {top_k}"
    results = list(conn.execute(sql))
    return [r[0] for r in results]


def get_top_k_vals_for_cols(
    table_name: str, columns: list[str], top_k: int = DEFAULT_TOP_K
) -> dict[str, list[str | bool | int | float]]:
    return {col: get_top_k_vals(table_name, col, top_k=top_k) for col in columns}


def build_chatgpt_system_ask_prompt(table_name: str) -> str:
    system_prompt = PromptBuilder()
    system_prompt.add_part(
        "intro", f"A SQL table named {table_name} is stored in the database with the following schema:"
    )
    system_prompt.add_part("schema", get_table_schema(table_name))
    system_prompt.add_part(
        "instruction",
        """Translate the following instruction into a SQL query on this table. Strong preference to return a single row with the answer. Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Only output a SQL query and nothing else. Do not apologize for your mistakes; just try again if presented with an error message.""",
    )
    return system_prompt.build()


def clean_gpt_sql_statement(stmt: str) -> str:
    stmt = stmt.replace("\n", " ").replace("\r", " ").replace("```", "")
    if not stmt.startswith("SELECT"):
        # trim apology :roll_eye:
        parts = stmt.partition("SELECT")
        stmt = parts[1] + parts[2]
    return stmt


def verify_and_attempt_fix_sql(
    sql_statement: str, chat_manager: ChatManager, remaining_tries: int = MAX_SQL_TRY_COUNT
) -> tuple[str, Cursor]:
    try:
        return sql_statement, conn.execute(sql_statement)
    except Exception as error:
        app.logger.info(f"Invalid SQL: {sql_statement}")
        app.logger.debug(str(error))
        if remaining_tries <= 0:
            raise error
        app.logger.info("Trying again...")
        chat_manager.add_message(
            "user", f"{str(error)}\nTry again. Do not apologize for the error. Output only the fixed SQL statement."
        )
        sql_statement = clean_gpt_sql_statement(chat_manager.prompt_ai())
        return verify_and_attempt_fix_sql(sql_statement, chat_manager, remaining_tries=remaining_tries - 1)


def build_ask_manager(table_name: str, user_question: str) -> ChatManager:
    """Returns a chat manager primed to prompt ai for the SQL query that can be used to answer user_question"""
    chat_manager = ChatManager()
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
    chat_manager = ChatManager()
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
        app.logger.info("Built Ask Manager.")
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

    tables = get_tables()
    tables.sort(key=lambda table: table.lower())
    return render_template(
        "index.html",
        answer=request.args.get("answer"),
        tables=tables,
        selected_table_name=request.args.get("selected_table_name"),
    )
