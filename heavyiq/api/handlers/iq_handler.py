from fastapi.concurrency import run_in_threadpool
from langsmith import Client
from starlette.exceptions import HTTPException as StarletteHTTPException

from heavyiq.api.models import (
    FeedbackRequest,
    FeedbackResponse,
    GenerateTableMetadataRequest,
    GenerateTableMetadataResponse,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
    AskHeavyAIDocsRequest,
    AskHeavyAIDocsResponse,
)
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import (
    NLtoAnswerChain,
    GenerateTableMetadataChain,
    get_nl_to_sql_chain_by_llm,
    get_ask_docs_chain,
)
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.langchain.logging import log_chain_call_async
from heavyiq.logging_utils import get_heavyiq_logger


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
    llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=0.0)
    chain_cls = get_nl_to_sql_chain_by_llm(llm=llm)
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=config.log_to_stdout)
    chain = chain_cls(llm=llm, database=db, callbacks=[file_callback_handler], tags=["rest-api", "query-endpoint"])  # type: ignore
    chain_input = {chain.input_key: request.question, "tables": request.tables}
    res = await log_chain_call_async(chain, chain_input, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return QueryResponse(
        sql=res[chain.output_key], sql_complexity=res[chain.output_complexity_key], feedback_id=feedback_id
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
    nl_sql_llm = await run_in_threadpool(get_llm_by_type, LLMType.NL_TO_SQL, temperature=0.0)
    nl_sql_chain = get_nl_to_sql_chain_by_llm(llm=nl_sql_llm)(
        llm=nl_sql_llm,
        database=db,
        callbacks=[file_callback_handler],
        tags=["rest-api", "question-endpoint"],
    )
    llm = await run_in_threadpool(get_llm_by_type, LLMType.SQL_TO_ANSWER, temperature=0.0)
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
        info=res[chain.output_fail_reason_key],
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
    await run_in_threadpool(
        langsmith_client.create_feedback,
        request.feedback_id,
        "user_feedback",
        score=request.score,
        comment=request.comment,
    )

    return FeedbackResponse()


async def handle_generate_table_metadata_async(
    request: GenerateTableMetadataRequest, db: HeavyDB
) -> GenerateTableMetadataResponse:
    """
    Handles /generate-table-metadata request.
    """
    llm = await run_in_threadpool(get_llm_by_type, LLMType.SQL_TO_ANSWER, temperature=0.0)
    chain = GenerateTableMetadataChain(llm=llm, database=db, tags=["rest-api", "generate-table-metadata-endpoint"])
    res = await log_chain_call_async(chain, request.table_name, "")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return GenerateTableMetadataResponse(
        table_name=request.table_name,
        summary=res[chain.output_summary_key],
        columns=res[chain.output_columns_key],
        feedback_id=feedback_id,
    )


async def handle_ask_heavyai_docs_async(request: AskHeavyAIDocsRequest) -> AskHeavyAIDocsResponse:
    """
    Handles /ask-heavyai-docs request.
    """
    from heavyiq.langchain.index.docs.create_index import create_heavyai_docs_index

    docs_index = create_heavyai_docs_index()
    chain = get_ask_docs_chain(docs_index, tags=["rest-api", "ask-heavyai-docs-endpoint"])
    res = await log_chain_call_async(chain, request.question, "", chain_name="ask-heavyai-docs")
    feedback_id = str(res["__run"].run_id) if "__run" in res else ""
    return AskHeavyAIDocsResponse(
        answer=res[chain.answer_key],
        sources=[source.strip() for source in res[chain.sources_answer_key].split(",")],
        feedback_id=feedback_id,
    )
