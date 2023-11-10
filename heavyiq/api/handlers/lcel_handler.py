from heavyiq.api.models import QueryRequest, QueryResponse
from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.heavydb import heavydb_context
from heavyiq.logging_utils import get_heavyiq_logger


async def handle_lcel_query_request(request: QueryRequest, db: HeavyDB) -> QueryResponse:
    """
    Async LCEL handler for /query request.
    """
    from heavyiq.controller.langchain import predict_sql_query_and_complexity_with_retries

    config, logger = get_config(), get_heavyiq_logger()
    file_callback_handler = logger.async_langchain_cb_handler(to_stdout=config.log_to_stdout)
    async with heavydb_context(db):  # type: ignore
        out = await predict_sql_query_and_complexity_with_retries(
            request.dict(), config={"callbacks": [file_callback_handler]}  # type: ignore
        )
        return QueryResponse(sql=out[0], sql_complexity=out[1], feedback_id="", logprobs={})
