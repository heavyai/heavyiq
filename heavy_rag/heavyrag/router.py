import logging

from fastapi import APIRouter
from pydantic import BaseModel

from heavyrag.main import get_heavydb_reader, get_or_sync_index, retrieve_index

router = APIRouter()


class QueryRequest(BaseModel):
    sessionid: str
    question: str


@router.post("/query")
def query(request: QueryRequest) -> dict:
    """
    Query the index.
    """
    dbreader = get_heavydb_reader(request.sessionid)
    index = get_or_sync_index(dbreader)
    tables_with_score = retrieve_index(index, request.question, dbname=dbreader.database_name)
    return tables_with_score
