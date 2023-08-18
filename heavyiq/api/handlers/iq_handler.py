import re
from typing import Any

from langsmith import Client
from starlette.exceptions import HTTPException as StarletteHTTPException

from heavyiq.api.models import (
    FeedbackRequest,
    FeedbackResponse,
    GenerateTableMetadataRequest,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
)
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import NLtoAnswerChain, get_nl_to_sql_chain_by_llm
from heavyiq.langchain.index.generate_table_documents import async_create_table_metadata
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.langchain.logging import log_chain_call, log_chain_call_async
from heavyiq.logging_utils import get_heavyiq_logger


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
    llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=0.0)
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    chain = chain_cls(llm=llm, database=db, tags=["rest-api", "query-endpoint"])  # type: ignore
    chain_input = {chain.input_key: question, "tables": tables}
    res = log_chain_call(chain, chain_input, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return QueryResponse(
        sql=res[chain.output_key], sql_complexity=res[chain.output_complexity_key], feedback_id=feedback_id
    )


async def handle_query_request_async(request: QueryRequest, db: HeavyDB) -> QueryResponse:
    """
    Async handler for /query request.

    Args:
        request (QueryRequest): request payload

    Returns:
        QueryResponse: response content
    """
    config = get_config()
    logger = get_heavyiq_logger()
    llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=0.0)
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=config.log_to_stdout)
    chain = chain_cls(llm=llm, database=db, callbacks=[file_callback_handler], tags=["rest-api", "query-endpoint"])  # type: ignore
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return QueryResponse(
        sql=res[chain.output_key], sql_complexity=res[chain.output_complexity_key], feedback_id=feedback_id
    )


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
    nl_sql_llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=0.0)
    nl_sql_chain = get_nl_to_sql_chain_by_llm(llm=nl_sql_llm)(
        llm=nl_sql_llm, database=db, tags=["rest-api", "question-endpoint"]
    )
    llm = get_llm_by_type(LLMType.SQL_TO_ANSWER, temperature=0.0)
    chain = NLtoAnswerChain(
        llm=llm,
        nl_sql_chain=nl_sql_chain,
        database=db,
        tags=["rest-api", "question-endpoint"],
    )
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = log_chain_call(chain, chain_input, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return QuestionResponse(
        answer=res[chain.output_key],
        sql=res[chain.output_sql_key],
        sql_complexity=res[chain.output_sql_complexity_key],
        feedback_id=feedback_id,
    )


async def handle_question_request_async(request: QuestionRequest, db: HeavyDB) -> QuestionResponse:
    """
    Handles /question request.

    Args:
        request (QuestionRequest): request payload

    Returns:
        QuestionResponse: response content
    """
    config = get_config()
    logger = get_heavyiq_logger()
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=config.log_to_stdout)
    nl_sql_llm = get_llm_by_type(LLMType.NL_TO_SQL, temperature=0.0)
    nl_sql_chain = get_nl_to_sql_chain_by_llm(llm=nl_sql_llm)(
        llm=nl_sql_llm,
        database=db,
        callbacks=[file_callback_handler],
        tags=["rest-api", "question-endpoint"],
    )
    llm = get_llm_by_type(LLMType.SQL_TO_ANSWER, temperature=0.0)
    chain = NLtoAnswerChain(
        llm=llm,
        nl_sql_chain=nl_sql_chain,
        database=db,
        callbacks=[file_callback_handler],
        tags=["rest-api", "question-endpoint"],
    )
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return QuestionResponse(
        answer=res[chain.output_key],
        sql=res[chain.output_sql_key],
        sql_complexity=res[chain.output_sql_complexity_key],
        feedback_id=feedback_id,
    )


async def handle_submit_feedback_request_async(request: FeedbackRequest) -> FeedbackResponse:
    """
    Handles /submit-feedback request.

    Args:
        request (FeedbackRequest): request payload

    Returns:
        FeedbackResponse: response content
    """
    from heavyiq.langchain.utils import is_langsmith_active

    if not is_langsmith_active:
        raise StarletteHTTPException(status_code=400, detail="Feedback is not enabled in the config.")
    langsmith_client = Client()
    langsmith_client.create_feedback(request.feedback_id, "user_feedback", score=request.score, comment=request.comment)
    return FeedbackResponse()


async def handle_generate_table_metadata_async(request: GenerateTableMetadataRequest, db: HeavyDB) -> dict[Any, Any]:
    """
    Handles /generate-table-metadata request.
    """
    table_metadata = await async_create_table_metadata(db, request.table_name)
    summary, column_description = re.split(r"(?m)\s*Column descriptions for.*", table_metadata)
    column_description_mapping = dict(re.findall(r"(?m)^\s*-\s*(?:\w+\.)?(\w+):\s*(.+)", column_description))
    return {"table_name": request.table_name, "summary": summary, "columns": column_description_mapping}
