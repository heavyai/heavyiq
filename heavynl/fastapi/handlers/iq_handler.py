from fastapi.concurrency import run_in_threadpool
from heavynl.config import get_config
from heavynl.logging_utils import get_heavynl_logger
from heavynl.fastapi.models import QueryRequest, QueryResponse, QuestionRequest, QuestionResponse
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import get_nl_to_sql_chain_by_llm, NLtoAnswerChain
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.langchain.logging import log_chain_call, log_chain_call_async
from heavynl.utils import strip_sql_comments


MODEL_NAME = get_config().openai_gpt_model


def handle_query_request(request: QueryRequest) -> QueryResponse:
    """
    Handles /query request.

    Args:
        request (QueryRequest): request payload

    Returns:
        QueryResponse: response content
    """
    tables = request.tables
    db = HeavyDB.from_session(request.session_id, include_tables=tables)
    question = request.question
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "query", "chain", "nl_to_sql_chain"])
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    chain = chain_cls(llm=llm, database=db, verbose=True)  # type: ignore
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_key])
    sql_complexity = db.complexity(sql)
    return QueryResponse(sql=sql, sql_complexity=sql_complexity)


async def handle_query_request_async(request: QueryRequest, db: HeavyDB) -> QueryResponse:
    """
    Async handler for /query request.

    Args:
        request (QueryRequest): request payload

    Returns:
        QueryResponse: response content
    """
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "query", "chain", "nl_to_sql_chain_async"])
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    chain = chain_cls(llm=llm, database=db, verbose=True)  # type: ignore
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_key])
    sql_complexity = await run_in_threadpool(db.complexity, sql)
    return QueryResponse(sql=sql, sql_complexity=sql_complexity)


def handle_question_request(request: QuestionRequest) -> QuestionResponse:
    """
    Handles /question request.

    Args:
        request (QuestionRequest): request payload

    Returns:
        QuestionResponse: response content
    """
    logger = get_heavynl_logger()
    db = HeavyDB.from_session(request.session_id, include_tables=request.tables)
    logger.info("Request Received")
    logger.info("Table(s): %s", ", ".join(request.tables))
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "question", "chain", "nl_to_answer_chain"])
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = log_chain_call(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_sql_key])
    sql_complexity = db.complexity(sql)
    return QuestionResponse(answer=res[chain.output_answer_key], sql=sql, sql_complexity=sql_complexity)


async def handle_question_request_async(request: QuestionRequest, db: HeavyDB) -> QuestionResponse:
    """
    Handles /question request.

    Args:
        request (QuestionRequest): request payload

    Returns:
        QuestionResponse: response content
    """
    llm = get_llm_by_model_name(MODEL_NAME, ["rest_api", "question", "chain", "nl_to_answer_chain_async"])
    chain = NLtoAnswerChain(llm=llm, database=db, verbose=True)
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, MODEL_NAME)
    sql = strip_sql_comments(res[chain.output_sql_key])
    sql_complexity = await run_in_threadpool(db.complexity, sql)
    return QuestionResponse(answer=res[chain.output_answer_key], sql=sql, sql_complexity=sql_complexity)
