import asyncio
from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore

from heavyiq.langchain.heavydb import HeavyDB
from heavyrag import IndexNotFound
from heavyrag.evaluate import aevaluate_response_by_relevancy
from heavyrag.filters import document_filters, get_document_filter_matches, get_facts_filter_matches, table_filters
from heavyrag.index import get_index
from heavyrag.ingest import sync_table_index
from heavyrag.llm import get_llm
from heavyrag.prompts import FACTS_QA_PROMPT, TEXT_QA_PROMPT


async def ask_document(
    question: str, heavydb_name: str, file_name: str | None = None, do_evaluate: bool = True
) -> tuple[RESPONSE_TYPE, EvaluationResult | None]:
    """
    Ask questions about a document or a group of documents mapped to a particular database.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        raise IndexNotFound(f"VectorStoreIndex not found {heavydb_name} collection.")

    filters = None
    if file_name:
        filters = get_document_filter_matches(file_name=file_name)

    engine = index.as_query_engine(
        llm=get_llm(),
        filters=filters,
        similarity_top_k=2,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=TEXT_QA_PROMPT,
    )
    response, eval_result = await engine.aquery(question), None
    if do_evaluate:
        eval_result = await aevaluate_response_by_relevancy(question=question, response=response)

    return response, eval_result


async def retrieve_facts(
    index: VectorStoreIndex, question: str, heavydb_name: str, similarity_cutoff: float = 0.30
) -> list[NodeWithScore]:
    """
    Retrieve only the fact nodes which are relevant to the asked question without using llm.
    """
    doc_engine = index.as_retriever(
        filters=document_filters,
        similarity_top_k=2,
    )
    facts_engine = index.as_retriever(
        filters=get_facts_filter_matches(heavydb_name),
        similarity_top_k=2,
    )
    doc_nodes_with_score, facts_nodes_with_score = await asyncio.gather(
        doc_engine.aretrieve(question), facts_engine.aretrieve(question)
    )
    final_nodes = facts_nodes_with_score + doc_nodes_with_score
    # filter nodes below 0.30 similarity score
    processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    filtered_nodes = processor.postprocess_nodes(final_nodes)
    return filtered_nodes


async def get_relevant_facts_info(question: str, heavydb_name: str, similarity_cutoff: float = 0.30) -> str:
    """
    Function which supposed to return facts information relevant to the asked question by querying the index.
    """
    facts_info = ""
    retrieved_facts = await ask_facts(
        question=question, heavydb_name=heavydb_name, only_retrieve=True, similarity_cutoff=similarity_cutoff
    )
    for fact, metadata in retrieved_facts:
        facts_info += fact + "\n"

    return facts_info


async def ask_facts(
    question: str,
    heavydb_name: str,
    only_retrieve: bool = True,
    do_evaluate: bool = False,
    metada_mode: Literal["all", "llm", "none"] = "none",
    similarity_cutoff: float = 0.30,
) -> tuple[RESPONSE_TYPE, EvaluationResult | None] | list[tuple[str, dict]]:
    """
    Search document, facts nodes and retrieve facts relevant to the asked question.

    Params:
    - only_retrieve : If enabled then it will do only the RAG retrieval (no call to llm)
    - do_evaluate: If enabled, it should evaluate the final answer to see whether it's a proper fact or not.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        raise IndexNotFound(f"VectorStoreIndex not found {heavydb_name} collection.")

    if only_retrieve:
        filtered_nodes = await retrieve_facts(
            index, question=question, heavydb_name=heavydb_name, similarity_cutoff=similarity_cutoff
        )
        return [(node.get_content(metadata_mode=metada_mode), node.metadata) for node in filtered_nodes]

    engine = index.as_query_engine(
        llm=get_llm(),
        filters=document_filters,
        similarity_top_k=2,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=FACTS_QA_PROMPT,
    )
    response, eval_result = await engine.aquery(question), None
    if do_evaluate:
        eval_result = await aevaluate_response_by_relevancy(question=question, response=response)

    return response, eval_result


async def determine_table_names(question: str, heavydb: HeavyDB, force_sync: bool = False) -> list[str]:
    """
    Fetch the approprate table name relevant to the asked question.
    """
    index = await sync_table_index(heavydb=heavydb, force_sync=force_sync)
    engine = index.as_retriever(
        filters=table_filters,
        similarity_top_k=2,
    )
    nodes_with_score = await engine.aretrieve(question)
    tables = []
    for n in nodes_with_score:
        table_name = n.metadata["name"]
        if table_name not in tables:
            tables.append(table_name)

    return tables
