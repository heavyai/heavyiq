from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from heavynl.api.utils import handle_errors
from heavynl.logging_utils import default_logger as logger
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import NLtoSQLChain, NLtoAnswerChain
from heavynl.langchain.llms import get_chat_llm, get_llm
from heavynl.langchain.logging import log_chain_call
from heavynl.langchain.index import create_and_write_table_document, get_heavydb_index
from heavynl.utils import strip_sql_comments


MODEL_NAME = "gpt-4"


def build_llm(model: str, tags: list[str] = []) -> ChatOpenAI | OpenAI:
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        return get_chat_llm(tags, model_name=model)
    else:
        return get_llm(tags, temperature=0.2, model_name=model)


def parse_tables_from_body(body: dict, heavydb: HeavyDB) -> list[str]:
    if "tables" not in body or len(body["tables"]) == 0:
        allowable_tables = list(heavydb.get_usable_table_names())
        # there are options here
        # tables = get_heavydb_index().ask_using_rephrased_question(body["question"], allowable_tables=allowable_tables)["tables"]
        # tables = get_heavydb_index().ask_about_database(body["question"], allowable_tables=allowable_tables)["tables"]
        tables = get_heavydb_index().simple_search_for_table_names(body["question"], allowable_tables=allowable_tables)
    else:
        tables = body["tables"]
    return tables


@handle_errors
def query(body: dict) -> dict:
    # db_session_id = args["session_id"]
    # db = HeavyDB.from_session(db_session_id, include_tables=tables)
    db = HeavyDB.from_env()
    tables = parse_tables_from_body(body, db)
    question = body["question"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = build_llm(MODEL_NAME, ["rest_api", "query", "chain", "nl_to_sql_chain"])
    chain = NLtoSQLChain(llm=llm, database=db, verbose=True)
    input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, input, MODEL_NAME)
    response = {"sql": strip_sql_comments(res[chain.output_key])}
    return response


@handle_errors
def question(body: dict) -> dict:
    # db_session_id = args["session_id"]
    # db = HeavyDB.from_session(db_session_id)
    db = HeavyDB.from_env()
    tables = parse_tables_from_body(body, db)
    question = body["question"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = build_llm(MODEL_NAME, ["rest_api", "question", "chain", "nl_to_answer_chain"])
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, input, MODEL_NAME)
    return {"answer": res[chain.output_answer_key], "sql": strip_sql_comments(res[chain.output_sql_key])}


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
