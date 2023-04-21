from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.api.utils import handle_errors
from modules.logging_utils import default_logger as logger
from modules.langchain import HeavyDB
from modules.langchain.chains import NLtoSQLChain, NLtoAnswerChain
from modules.langchain.logging import log_chain_call
from modules.langchain.index import create_and_write_table_document, get_heavydb_index
from modules.utils import strip_sql_comments


MODEL_NAME = "gpt-4"


def build_llm(model: str = MODEL_NAME) -> ChatOpenAI | OpenAI:
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        return ChatOpenAI(model_name=model, client=None)
    else:
        return OpenAI(
            model_name=model,
            client=None,
            temperature=0.2,
        )


def parse_tables_from_body(body: dict) -> list[str]:
    if "tables" not in body or len(body["tables"]) == 0:
        # there are options here
        # tables = get_heavydb_index().ask_using_rephrased_question(body["question"])["tables"]
        # tables = get_heavydb_index().ask_about_database(body["question"])["tables"]
        tables = get_heavydb_index().simple_search_for_table_names(body["question"])
    else:
        tables = body["tables"]
    return tables


@handle_errors
def query(body: dict) -> dict:
    tables = parse_tables_from_body(body)
    question = body["question"]
    # db_session_id = args["session_id"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = build_llm()
    # db = HeavyDB.from_session(db_session_id, include_tables=tables)
    db = HeavyDB.from_env(include_tables=tables)
    chain = NLtoSQLChain(llm=llm, database=db, verbose=True)
    input = {"query": question, "table_names_to_use": tables}
    res = log_chain_call(chain, input, MODEL_NAME)
    response = {"sql": strip_sql_comments(res[chain.output_key])}
    return response


@handle_errors
def question(body: dict) -> dict:
    tables = parse_tables_from_body(body)
    question = body["question"]
    # db_session_id = args["session_id"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = build_llm()
    # db = HeavyDB.from_session(db_session_id, include_tables=tables)
    db = HeavyDB.from_env()
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    input = {"query": question, "tables": tables}
    res = log_chain_call(chain, input, MODEL_NAME)
    return {"answer": res[chain.output_answer_key], "sql": strip_sql_comments(res[chain.output_key])}


@handle_errors
def add_tables(body: dict) -> dict:
    tables = body["tables"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    # db = HeavyDB.from_session(db_session_id, include_tables=tables)
    db = HeavyDB.from_env(include_tables=tables)
    process_func = partial(create_and_write_table_document, db)
    with ThreadPoolExecutor() as executor:
        executor.map(process_func, tables)
    logger.info("Finished generating summaries")
    heavydb_index = get_heavydb_index()
    for table in tables:
        heavydb_index.reindex_table_document(table)
    return {"status": "Tables successfully added to Metadata Index"}
