#############
# Chain/Runnable which is supposed to RAG retrieve relevant facts
#############
import json
from typing import Any

from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from pydantic import BaseModel, Field

from heavyiq.lcel.callbacks.file_callback import LogFileCallbackHandler
from heavyiq.logging_utils import get_heavyrag_logger
from heavyrag.filters import get_facts_filter_matches
from heavyrag.postprocessor import ReRanker, ReRankerType
from heavyrag.utils import get_facts_node_count_in_index

rag_logger = get_heavyrag_logger()


class RetrieveFactsInputType(BaseModel):
    """
    Retrieve Facts input type.
    """

    question: str = Field(..., description="Actual question.")
    collection_name: str = Field(
        ..., description="Name of the collection (ie. heavydb name) on which RAG is supposed to happen."
    )
    similarity_cutoff: float = Field(
        default=0.30,
        description="Specifies the similarity score threshold that a document must meet or exceed to be included in the results. ",
    )
    similarity_top_k: int = Field(
        default=5, description="Specifies the number of top similar documents to retrieve from the index. "
    )
    use_reranker: bool = Field(default=True, description="Whether to enable reranker or not.")
    reranker_cutoff: float = Field(
        default=0.1,
        description="Sets a threshold for the reranker's scoring. Documents with scores below this cutoff will be excluded from the final output.",
    )
    reranker_top_k: int = Field(
        default=3, description="Specifies the maximum number of documents to be considered by the reranker"
    )


def fetch_index(inputs: dict) -> VectorStoreIndex | IndexError:
    """
    Get index by heavydb name.
    """
    from heavyrag.index import get_index

    collection_name = inputs["collection_name"]
    index = get_index(collection_name=collection_name)
    if not index:
        None
    return index


async def calculate_available_snippets_count(inputs: dict) -> int:
    return await get_facts_node_count_in_index(inputs["index"])


async def retrieve_relevant_snippets(inputs: dict) -> list[Any]:
    """
    Supposed to retrieve snippets relevant to the asked question.
    """
    index, collection_name, similarity_top_k, question = (
        inputs["index"],
        inputs["collection_name"],
        inputs["similarity_top_k"],
        inputs["question"],
    )
    facts_engine = index.as_retriever(
        filters=get_facts_filter_matches(collection_name),
        similarity_top_k=similarity_top_k,
    )
    try:
        return await facts_engine.aretrieve(question)
    except json.JSONDecodeError:
        msg = 'Invalid response from Embeddings API: check embeddings server ("rag_embed_server_base") is reachable or not.'
        rag_logger.error(msg)
        raise ValueError(msg)


async def retrieve_and_filter_snippets(inputs: dict) -> list[Any]:
    """
    Retrieve and filter snippets by similarity cutoff.
    """
    nodes = await retrieve_relevant_snippets(inputs)
    processor = SimilarityPostprocessor(similarity_cutoff=inputs["similarity_cutoff"])
    filtered_nodes = processor.postprocess_nodes(nodes)
    return filtered_nodes


async def apply_reranker_and_filter_top_n(inputs: dict) -> list[Any]:
    """
    Apply node reranking (ie. sort nodes) and filter the top-n nodes.
    """
    reranker = ReRanker(
        reranker_type=ReRankerType.OverridedLLMReRanker,
        top_n=inputs["reranker_top_k"],
        cutoff_score=inputs["reranker_cutoff"],
    )
    reranked_nodes = await reranker._apostprocess_nodes(inputs["nodes"], query_str=inputs["question"])
    return reranked_nodes


async def rerank_nodes(inputs: dict) -> dict:
    """
    Rerank nodes.
    """
    reranker = ReRanker(
        reranker_type=ReRankerType.OverridedLLMReRanker,
        top_n=inputs["reranker_top_k"],
        cutoff_score=inputs["reranker_cutoff"],
    )
    reranked_nodes = await reranker._rerank_nodes(inputs["nodes"], query_str=inputs["question"])
    return {"reranked_nodes": reranked_nodes, "reranker_cutoff": inputs["reranker_cutoff"]}


def filter_nodes_by_reranker_cutoff(inputs: dict) -> list[Any]:
    """
    Filter nodes by reranker cutoff.
    """
    from llama_index.core.postprocessor import SimilarityPostprocessor

    processor = SimilarityPostprocessor(similarity_cutoff=inputs["reranker_cutoff"])
    filtered_nodes = processor.postprocess_nodes(inputs["reranked_nodes"])
    return filtered_nodes


async def fetch_and_filter_snippets_by_reranker(inputs: dict) -> list[Any]:
    """
    Fetch all database-related snippets from RAGDB and pass them to the reranker to filter out the relevant snippets.
    """
    from heavyrag.main import filter_snippets_using_reranker, list_database_facts

    facts = list_database_facts(database_name=inputs["collection_name"])
    nodes = await filter_snippets_using_reranker(
        question=inputs["question"],
        snippets=facts,
        reranker_top_k=inputs["reranker_top_k"],
        reranker_cutoff=inputs["reranker_cutoff"],
    )
    return nodes


rerank_filter_runnable: Runnable = RunnableLambda(rerank_nodes) | RunnableLambda(filter_nodes_by_reranker_cutoff)  # type: ignore
rerank_snippets_chain: Runnable = RunnablePassthrough.assign(nodes=rerank_filter_runnable)

retrieve_snippets_chain: Runnable = RunnablePassthrough.assign(nodes=retrieve_and_filter_snippets) | RunnableBranch(
    (lambda x: x["use_reranker"], rerank_snippets_chain), lambda x: x
)

has_index_chain: Runnable = (
    RunnablePassthrough.assign(available_snippets_count=calculate_available_snippets_count)
    | RunnableBranch((lambda x: x["available_snippets_count"] > 0, retrieve_snippets_chain), lambda x: None)
    | RunnableLambda(lambda x: x.get("nodes") if x else [])
)

snippet_nodes_chain: Runnable = (
    RunnablePassthrough.assign(index=fetch_index)
    | RunnableBranch((lambda x: x["index"], has_index_chain), fetch_and_filter_snippets_by_reranker)
).with_types(
    input_type=RetrieveFactsInputType  # type: ignore
)  # return list[NodeWithScore]

snippets_chain: Runnable = snippet_nodes_chain | RunnableLambda(lambda x: [(i.node_id, i.get_text()) for i in x])

# Adding callbacks to the chain enables automatic logging when the chain is invoked
chain = snippets_chain.with_config(callbacks=[LogFileCallbackHandler(logger=rag_logger)])  # type: ignore
# Output Syntax: [(<node_id_1>, <node_text_1>), (<node_id_2>, <node_text_2>)], Example [('w1232e3dedscdqwecd', 'Node content')]
