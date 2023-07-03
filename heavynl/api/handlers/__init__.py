from heavynl.api.utils import handle_errors
from heavynl.config import get_config
from heavynl.logging_utils import get_heavynl_logger
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import NLtoAnswerChain, get_nl_to_sql_chain_by_llm
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.langchain.logging import log_chain_call
from heavynl.utils import strip_sql_comments


MODEL_NAME = get_config().openai_gpt_model


@handle_errors
def query(body: dict) -> dict:
    logger = get_heavynl_logger()
    db_session_id = body["session_id"]
    tables = body["tables"]
    db = HeavyDB.from_session(db_session_id, include_tables=tables)
    question = body["question"]
    logger.info("Request Received")
    logger.info("Table(s): %s", ", ".join(tables))
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "query", "chain", "nl_to_sql_chain"])
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    chain = chain_cls(llm=llm, database=db, verbose=True)  # type: ignore
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_key])
    sql_complexity = db.complexity(sql)
    response = {"sql": sql, "sql_complexity": sql_complexity}
    return response


@handle_errors
def question(body: dict) -> dict:
    logger = get_heavynl_logger()
    db_session_id = body["session_id"]
    tables = body["tables"]
    db = HeavyDB.from_session(db_session_id, include_tables=tables)
    question = body["question"]
    logger.info("Request Received")
    logger.info("Table(s): %s", ", ".join(tables))
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "question", "chain", "nl_to_answer_chain"])
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_sql_key])
    sql_complexity = db.complexity(sql)
    return {"answer": res[chain.output_answer_key], "sql": sql, "sql_complexity": sql_complexity}


@handle_errors
def health():
    return


# @handle_errors
# def add_tables(body: dict) -> dict:
#    tables = body["tables"]
#    # dbname = body["databaseName"]
#    logger.info("Request Received")
#    logger.info(f"Table(s): {', '.join(tables)}")
#    # db = HeavyDB.from_session(db_session_id, dbname=dbname, include_tables=tables)
#    db = HeavyDB.from_env(include_tables=tables)
#    process_func = partial(create_and_write_table_document, db)
#    with ThreadPoolExecutor() as executor:
#        executor.map(process_func, tables)
#    logger.info("Finished generating summaries")
#    heavydb_index = get_heavydb_index()
#    for table in tables:
#        heavydb_index.reindex_table_document(table)
#    return {"status": "Tables successfully added to Metadata Index"}
