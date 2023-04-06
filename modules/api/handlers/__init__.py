from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from modules.api.utils import handle_errors
from modules.logging_utils import default_logger as logger
from modules.langchain import HeavyDB
from modules.langchain.chains.heavydb.base import NLtoSQLChain, NLtoAnswerChain
from modules.langchain.logging import log_chain_call


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


@handle_errors
def query(body: dict) -> dict:
    tables = body["tables"]
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
    response = {"sql": res[chain.output_key]}
    return response


@handle_errors
def question(body: dict) -> dict:
    tables = body["tables"]
    question = body["question"]
    # db_session_id = args["session_id"]
    logger.info("Request Received")
    logger.info(f"Table(s): {', '.join(tables)}")
    llm = build_llm()
    # db = HeavyDB.from_session(db_session_id, include_tables=tables)
    db = HeavyDB.from_env()
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    input = {"query": question, "table_names_to_use": tables}
    res = log_chain_call(chain, input, MODEL_NAME)
    return {"answer": res[chain.output_answer_key], "sql": res[chain.output_sql_key]}
