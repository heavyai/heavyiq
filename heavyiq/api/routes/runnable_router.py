from fastapi import APIRouter
from langserve import add_routes

from heavyiq.lcel.chains.heavydb.sql_chain import chain as sql_chain_runnable

runnable_router = APIRouter()
# add langserve apis
add_routes(runnable_router, sql_chain_runnable, path="/lcel/nl-to-sql")
