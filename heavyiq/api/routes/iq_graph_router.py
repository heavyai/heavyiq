from typing import Any

from fastapi import APIRouter

from heavyiq.api.handlers.graph_handler import (
    handle_graph_auto_query_request,
    handle_graph_auto_question_request,
    handle_graph_query_request,
    handle_graph_question_request,
)
from heavyiq.api.models import (
    AutoQueryRequest,
    AutoQueryResponse,
    AutoQuestionRequest,
    AutoQuestionResponse,
    QueryRequest,
    QueryResponse,
    QuestionRequest,
    QuestionResponse,
)
from heavyiq.api.routes.log_route import LoggingRoute

graph_router = APIRouter(route_class=LoggingRoute)


@graph_router.post("/query", response_model=QueryResponse)
async def query(values: QueryRequest) -> QueryResponse:
    """
    Request for sql query with all the information using LangGraph.
    """
    return await handle_graph_query_request(values)


@graph_router.post("/auto/query", response_model=AutoQueryResponse)
async def auto_query(values: AutoQueryRequest) -> AutoQueryResponse:
    """
    Request for auto sql query with all the information using LangGraph.
    """
    return await handle_graph_auto_query_request(values)


@graph_router.post("/question", response_model=QuestionResponse)
async def question(values: QuestionRequest) -> QuestionResponse:
    """
    Request for auto sql query with all the information using LangGraph.
    """
    return await handle_graph_question_request(values)


@graph_router.post("/auto/question", response_model=AutoQuestionResponse)
async def auto_question(values: AutoQuestionRequest) -> AutoQuestionResponse:
    """
    Request for auto sql query with all the information using LangGraph.
    """
    return await handle_graph_auto_question_request(values)
