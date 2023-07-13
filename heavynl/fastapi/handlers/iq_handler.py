from heavynl.config import get_config
from heavynl.fastapi.models import QueryRequest, QueryResponse
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import get_nl_to_sql_chain_by_llm
from heavynl.langchain.llms import get_llm_by_model_name
from heavynl.langchain.logging import log_chain_call
from heavynl.utils import strip_sql_comments


MODEL_NAME = get_config().openai_gpt_model


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
