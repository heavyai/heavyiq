from fastapi import APIRouter
from langserve import add_routes

from heavyiq.lcel.chains import answer_chain, auto_answer_chain, auto_sql_chain, sql_chain, table_chain

runnable_router = APIRouter()
# add langserve apis
add_routes(runnable_router, sql_chain, path="/lcel/nl-to-sql")
add_routes(runnable_router, answer_chain, path="/lcel/nl-to-answer")
add_routes(runnable_router, table_chain, path="/lcel/nl-to-tables")
add_routes(runnable_router, auto_sql_chain, path="/lcel/auto/nl-to-sql")
add_routes(runnable_router, auto_answer_chain, path="/lcel/auto/nl-to-answer")
