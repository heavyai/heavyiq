import asyncio

from async_lru import alru_cache
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
from heavyrag.ingest import sync_table_index
from heavyrag.llm import get_llm
from heavyrag.logger import logger
from heavyrag.postprocessor import OverridedLLMRerank
from heavyrag.prompts import FACTS_QA_PROMPT, TEXT_QA_PROMPT
from heavyrag.utils import get_document_node_count_in_index, get_facts_node_count_in_index, get_nodes


def get_index(collection_name: str) -> VectorStoreIndex:
    """
    Get Vecstor Store Index.
    """
    from heavyrag.controller import rag_controller

    return rag_controller.index(collection_name)


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
    index: VectorStoreIndex,
    question: str,
    heavydb_name: str,
    similarity_cutoff: float = 0.30,
    similarity_top_k: int = 5,
    with_reranker: bool = False,
    reranker_top_k: int = 3,
    reranker_cutoff: float = 0.01,
) -> list[NodeWithScore]:
    """
    Retrieve only the fact nodes which are relevant to the asked question without using llm.
    """
    from heavyiq.config import get_config

    config = get_config()

    logger.debug("Started retrieving facts...")

    doc_nodes_with_score, facts_nodes_with_score = [], []
    doc_node_count, facts_node_count = await asyncio.gather(
        get_document_node_count_in_index(index), get_facts_node_count_in_index(index)
    )
    logger.debug(f"Snippets node count: {facts_node_count}, Document node count: {doc_node_count}")
    if doc_node_count > 0:
        doc_engine = index.as_retriever(
            filters=document_filters,
            similarity_top_k=similarity_top_k,
        )
        doc_nodes_with_score = await doc_engine.aretrieve(question)

    if facts_node_count > 0:
        facts_engine = index.as_retriever(
            filters=get_facts_filter_matches(heavydb_name),
            similarity_top_k=similarity_top_k,
        )
        facts_nodes_with_score = await facts_engine.aretrieve(question)

    final_nodes = facts_nodes_with_score + doc_nodes_with_score
    logger.debug(f"Nodes found after applying facts and document filters, {[(i.id_, i.score) for i in final_nodes]}")
    # filter nodes below 0.30 similarity score
    processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    filtered_nodes = processor.postprocess_nodes(final_nodes)
    logger.debug(
        f"Filtered nodes after applying similarity postprocessor with similarity_cutoff {similarity_cutoff}: {[(i.id_, i.score) for i in filtered_nodes]}"
    )
    if with_reranker:
        logger.debug("Started re-ranking filtered nodes...")
        # configure reranker
        # reranker = TextEmbeddingsInferenceRerank(
        #     base_url=config.rag_rerank_server_base,
        #     top_n=reranker_top_k,
        # )
        # filtered_nodes = await reranker.apostprocess_nodes(filtered_nodes, query_str=question)
        reranker = OverridedLLMRerank(top_n=reranker_top_k)
        filtered_nodes = reranker.postprocess_nodes(filtered_nodes, query_str=question)
        logger.debug(
            f"Filtered nodes after applying ReRanking postprocessor: {[(i.id_, i.score) for i in filtered_nodes]}"
        )
        # apply postprocessor to drop off nodes which has the score lesser than the reranker_cutoff
        filtered_nodes = SimilarityPostprocessor(similarity_cutoff=reranker_cutoff).postprocess_nodes(filtered_nodes)
        logger.debug(
            f"Filtered nodes after applying similarity postprocessor with reranker_cutoff {reranker_cutoff}: {[(i.id_, i.score) for i in filtered_nodes]}"
        )

    return filtered_nodes


async def get_relevant_facts_info(
    question: str,
    heavydb_name: str,
    similarity_cutoff: float = 0.30,
    similarity_top_k: int = 5,
    with_reranker: bool = False,
    reranker_top_k: int = 3,
    reranker_cutoff: float = 0.01,
) -> str:
    """
    Function which supposed to return facts information relevant to the asked question by querying the index.
    """
    facts_info = ""
    logger.debug(f"Getting relevant facts from {heavydb_name} collection for question: {question}...")
    try:
        retrieved_facts = await ask_facts(
            question=question,
            heavydb_name=heavydb_name,
            only_retrieve=True,
            similarity_cutoff=similarity_cutoff,
            similarity_top_k=similarity_top_k,
            with_reranker=with_reranker,
            reranker_top_k=reranker_top_k,
            reranker_cutoff=reranker_cutoff,
        )
        for fact, metadata, score in retrieved_facts:
            facts_info += fact + "\n\n"

        facts_info = facts_info.strip()
        loginfo = f"Facts found? {True if facts_info else False}"
        if facts_info:
            loginfo += "\n" + facts_info
        logger.debug(loginfo)
    except IndexNotFound:
        logger.info(f"Relevant Collection {heavydb_name} does not exist!")
    return facts_info


@alru_cache(ttl=10)
async def get_relevant_facts_info_from_cache(
    question: str,
    heavydb_name: str,
    similarity_cutoff: float = 0.30,
    similarity_top_k: int = 5,
    with_reranker: bool = False,
    reranker_top_k: int = 3,
    reranker_cutoff: float = 0.01,
) -> str:
    """
    Get the relevant facts from cache or calculate.
    """
    return await get_relevant_facts_info(
        question=question,
        heavydb_name=heavydb_name,
        similarity_cutoff=similarity_cutoff,
        similarity_top_k=similarity_top_k,
        with_reranker=with_reranker,
        reranker_top_k=reranker_top_k,
        reranker_cutoff=reranker_cutoff,
    )


async def get_facts(heavydb_name: str, limit: int = 100) -> list[NodeWithScore]:
    """
    Get all the fact nodes without querying the vectorDB.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        raise IndexNotFound(f"VectorStoreIndex not found {heavydb_name} collection.")
    nodes = await get_nodes(index=index, filters=get_facts_filter_matches(heavydb_name), limit=limit)
    return nodes


async def ask_facts(
    question: str,
    heavydb_name: str,
    only_retrieve: bool = True,
    do_evaluate: bool = False,
    similarity_cutoff: float = 0.30,
    similarity_top_k: int = 5,
    with_reranker: bool = False,
    reranker_top_k: int = 3,
    reranker_cutoff: float = 0.01,
) -> tuple[RESPONSE_TYPE, EvaluationResult | None] | list[tuple[str, dict, float | None]]:
    """
    Search document, facts nodes and retrieve facts relevant to the asked question.

    Params:
    - only_retrieve : If enabled then it will do only the RAG retrieval (no call to llm)
    - do_evaluate: If enabled, it should evaluate the final answer to see whether it's a proper fact or not.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        raise IndexNotFound(f"VectorStoreIndex not found {heavydb_name} collection.")

    logger.debug(
        "Called ask_facts with args:\n"
        f"question: {question}\n"
        f"heavydb_name: {heavydb_name}\n"
        f"only_retrieve: {only_retrieve}\n"
        f"do_evaluate: {do_evaluate}\n"
        f"similarity_cutoff: {similarity_cutoff}\n"
        f"similarity_top_k: {similarity_top_k}\n"
        f"with_reranker: {with_reranker}\n"
        f"reranker_top_k: {reranker_top_k}\n"
        f"reranker_cutoff: {reranker_cutoff}"
    )

    if only_retrieve:
        filtered_nodes = await retrieve_facts(
            index,
            question=question,
            heavydb_name=heavydb_name,
            similarity_cutoff=similarity_cutoff,
            similarity_top_k=similarity_top_k,
            with_reranker=with_reranker,
            reranker_top_k=reranker_top_k,
            reranker_cutoff=reranker_cutoff,
        )
        return [(node.get_text(), node.metadata, node.score) for node in filtered_nodes]

    logger.debug("Started generating response using LLM from the contents of retrieved nodes...")
    engine = index.as_query_engine(
        llm=get_llm(),
        filters=document_filters,
        similarity_top_k=similarity_top_k,
        similarity_cutoff=similarity_cutoff,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=FACTS_QA_PROMPT,
    )
    response, eval_result = await engine.aquery(question), None
    if do_evaluate:
        logger.debug("Evaluating the RAG LLM response...")
        eval_result = await aevaluate_response_by_relevancy(question=question, response=response)

    return response, eval_result


async def determine_table_names(question: str, heavydb: HeavyDB, force_sync: bool = False) -> list[str]:
    """
    Fetch the approprate table name relevant to the asked question.
    """
    logger.debug("Started determining table names, syncing table index...")
    from heavyrag.controller import rag_controller

    await rag_controller.sync_table_nodes(heavydb=heavydb)
    logger.debug("Finished syncing table index.")
    index = rag_controller.index(heavydb._dbname)
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

    logger.debug("Retrieved tables: {tables}")

    return tables
