from fastapi import APIRouter
from langserve import add_routes

from heavyiq.lcel.chains import answer_chain, sql_chain

runnable_router = APIRouter()
# add langserve apis
add_routes(runnable_router, sql_chain, path="/lcel/nl-to-sql")
add_routes(runnable_router, answer_chain, path="/lcel/nl-to-answer")
