from langchain.schema.runnable.config import RunnableConfig

from heavyiq.langchain.exceptions import NLtoSQLException
from heavyiq.lcel.chains.heavydb.sql_chain import DetachedComplexityChainInputDict
from heavyiq.lcel.chains.heavydb.sql_chain import SqlChainInputDict as PredictQueryInputDict
from heavyiq.lcel.chains.heavydb.sql_chain import SqlChainOutputDictWithComplexity as DetachedComplexityChainOutputDict
from heavyiq.lcel.chains.heavydb.sql_chain import chain as query_chain
from heavyiq.lcel.chains.heavydb.sql_chain import detached_complexity_chain
from heavyiq.lcel.chains.heavydb.sql_chain import retry_chain as query_retry_chain


async def predict_sql_query_with_retries(
    inputs: PredictQueryInputDict, config: RunnableConfig | None = None, max_retries: int = 3
) -> str:
    """
    Predict a functioning SQL query after several specific retries.

    Args:
        inputs: set of inputs passed to the underlying runnable.
        config: config passed to runnable.ainvoke coroutine. Defaults to None.
        max_retries: How many times to retry in-order to get a working SQL query.. Defaults to 3.

    Returns:
        A SQL query.
    """
    retries, verified, query = 0, False, None
    query = None  # Validated SQL query
    sql_cmd = None  # Recent SQL query which fails validation
    sql_chain, chain_inputs = query_chain, inputs
    while not verified:
        try:
            query = await sql_chain.ainvoke(chain_inputs, config=config)
            verified = True
        except NLtoSQLException as e:
            retries += 1
            if retries > max_retries:
                break
            extracted_error, sql_cmd = e.message, e.failed_sql
            error_inputs = {
                **inputs,
                "error": extracted_error[0:150] if len(extracted_error) > 150 else extracted_error,
                "sql_cmd": sql_cmd,
            }
            sql_chain, chain_inputs = query_retry_chain, error_inputs  # type: ignore

    if not verified:
        raise NLtoSQLException(
            f"Language model failed to generate a valid SQL query after {max_retries} tries.", sql_cmd
        )

    return query


async def predict_sql_query_and_complexity_with_retries(
    inputs: PredictQueryInputDict, config: RunnableConfig | None = None, max_retries: int = 3
) -> tuple[str, int]:
    """
    Predict a functioning SQL query after several specific retries and also find it's complexity.

    Args:
        inputs: set of inputs passed to the underlying runnable.
        config: config passed to runnable.ainvoke coroutine. Defaults to None.
        max_retries: How many times to retry in-order to get a working SQL query.. Defaults to 3.

    Returns:
        A SQL query, sql_complexity.
    """
    query = await predict_sql_query_with_retries(inputs=inputs, config=config, max_retries=max_retries)
    chain_inputs: DetachedComplexityChainInputDict = {"inputs": inputs, "query": query}
    out: DetachedComplexityChainOutputDict = await detached_complexity_chain.ainvoke(chain_inputs)  # type: ignore
    return out["query"], out["sql_complexity"]
