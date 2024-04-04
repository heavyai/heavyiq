import logging

from fastapi import APIRouter
from pydantic import BaseModel

from heavyrag.main import ask_db_index

dbidx_router = APIRouter(prefix="/db")
docidx_router = APIRouter(prefix="/doc")


class DBIndexRetrieveRequest(BaseModel):
    sessionid: str
    question: str


@dbidx_router.post("/retrieve/tables")
def retrieve_tables(request: DBIndexRetrieveRequest) -> list[str]:
    """
    Retrieve
    """
    return ask_db_index(request.sessionid, request.question)
