from fastapi import APIRouter
from langserve import add_routes

# Lazy-load chains to ensure config is loaded first
# Chains are imported when get_runnable_router() is called, not at module import time

_runnable_router = None


def get_runnable_router() -> APIRouter:
    """
    Get the runnable router with langserve routes.
    Chains are imported lazily to ensure config is loaded before chain creation.
    """
    global _runnable_router
    if _runnable_router is None:
        from heavyiq.lcel.chains import answer_chain, auto_answer_chain, auto_sql_chain, sql_chain, table_chain
        
        _runnable_router = APIRouter()
        # add langserve apis
        add_routes(_runnable_router, sql_chain, path="/lcel/nl-to-sql")
        add_routes(_runnable_router, answer_chain, path="/lcel/nl-to-answer")
        add_routes(_runnable_router, table_chain, path="/lcel/nl-to-tables")
        add_routes(_runnable_router, auto_sql_chain, path="/lcel/auto/nl-to-sql")
        add_routes(_runnable_router, auto_answer_chain, path="/lcel/auto/nl-to-answer")
    
    return _runnable_router


# For backwards compatibility - but prefer using get_runnable_router()
runnable_router = APIRouter()
