from heavynl.api.utils import handle_errors
from heavynl.config import get_config
from heavynl.logging_utils import heavynl_logger as logger
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import NLtoSQLChain, NLtoAnswerChain
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.langchain.logging import log_chain_call
from heavynl.utils import strip_sql_comments


MODEL_NAME = get_config().openai_gpt_model


@handle_errors
def query(body: dict) -> dict:
    db_session_id = body["session_id"]
    tables = body["tables"]
    db = HeavyDB.from_session(db_session_id, include_tables=tables)
    question = body["question"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "query", "chain", "nl_to_sql_chain"])
    chain = NLtoSQLChain(llm=llm, database=db, verbose=True)
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    response = {"sql": strip_sql_comments(res[chain.output_key])}
    return response


@handle_errors
def question(body: dict) -> dict:
    db_session_id = body["session_id"]
    tables = body["tables"]
    db = HeavyDB.from_session(db_session_id, include_tables=tables)
    question = body["question"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "question", "chain", "nl_to_answer_chain"])
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    return {"answer": res[chain.output_answer_key], "sql": strip_sql_comments(res[chain.output_sql_key])}


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
