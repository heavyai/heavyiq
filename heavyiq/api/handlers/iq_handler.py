from fastapi.concurrency import run_in_threadpool

from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.api.models import QueryRequest, QueryResponse, QuestionRequest, QuestionResponse
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm, NLtoAnswerChain
from heavyiq.langchain.llms import get_llm_by_type, LLMType
from heavyiq.langchain.logging import log_chain_call, log_chain_call_async
from heavyiq.utils import strip_sql_comments


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
    llm = get_llm_by_type(LLMType.NL_TO_SQL, ["rest_api", "query", "chain", "nl_to_sql_chain"], temperature=0.0)
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    chain = chain_cls(llm=llm, database=db, verbose=True)  # type: ignore
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, "")
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
    logger = get_heavyiq_logger()
    llm = get_llm_by_type(LLMType.NL_TO_SQL, ["rest_api", "query", "chain", "nl_to_sql_chain"], temperature=0.0)
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=False)
    chain = chain_cls(llm=llm, database=db, callbacks=[file_callback_handler], verbose=True)  # type: ignore
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, "")
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
    logger = get_heavyiq_logger()
    db = HeavyDB.from_session(request.session_id, include_tables=request.tables)
    logger.info("Request Received")
    logger.info("Table(s): %s", ", ".join(request.tables))
    nl_sql_llm = get_llm_by_type(
        LLMType.NL_TO_SQL, ["rest_api", "question", "chain", "nl_to_sql_chain"], temperature=0.0
    )
    nl_sql_chain = get_nl_to_sql_chain_by_llm(llm=nl_sql_llm)(llm=nl_sql_llm, database=db, verbose=True)
    llm = get_llm_by_type(
        LLMType.SQL_TO_ANSWER, ["rest_api", "question", "chain", "nl_to_answer_chain"], temperature=0.0
    )
    chain = NLtoAnswerChain(llm=llm, nl_sql_chain=nl_sql_chain, database=db, verbose=True)
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = log_chain_call(chain, chain_input, "")
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
    logger = get_heavyiq_logger()
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=False)
    nl_sql_llm = get_llm_by_type(
        LLMType.NL_TO_SQL, ["rest_api", "question", "chain", "nl_to_sql_chain"], temperature=0.0
    )
    nl_sql_chain = get_nl_to_sql_chain_by_llm(llm=nl_sql_llm)(
        llm=nl_sql_llm, database=db, callbacks=[file_callback_handler], verbose=True
    )
    llm = get_llm_by_type(
        LLMType.SQL_TO_ANSWER, ["rest_api", "question", "chain", "nl_to_answer_chain"], temperature=0.0
    )
    chain = NLtoAnswerChain(
        llm=llm, nl_sql_chain=nl_sql_chain, database=db, callbacks=[file_callback_handler], verbose=True
    )
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, "")
    sql = strip_sql_comments(res[chain.output_sql_key])
    sql_complexity = await run_in_threadpool(db.complexity, sql)
    return QuestionResponse(answer=res[chain.output_answer_key], sql=sql, sql_complexity=sql_complexity)
