from heavyiq import CONFIG
from heavyiq.api.models import QueryRequest, QueryResponse
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.chains import get_nl_to_sql_chain_by_llm
from heavyiq.langchain.llms import get_llm_by_model_name
from heavyiq.langchain.logging import log_chain_call
from heavyiq.core.utils import strip_sql_comments


MODEL_NAME = CONFIG.openai_gpt_model


def handle_query_request(request: QueryRequest):
    # Request handling logic goes here
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
