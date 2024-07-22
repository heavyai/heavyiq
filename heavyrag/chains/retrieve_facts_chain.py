#############
# Chain/Runnable which is supposed to RAG retrieve relevant facts
#############
from typing import Any

from langchain_core.beta.runnables.context import Context
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
    return await facts_engine.aretrieve(question)


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


rerank_snippets_chain: Runnable = RunnablePassthrough.assign(nodes=apply_reranker_and_filter_top_n)

retrieve_snippets_chain: Runnable = RunnablePassthrough.assign(nodes=retrieve_and_filter_snippets) | RunnableBranch(
    (lambda x: x["use_reranker"], rerank_snippets_chain), lambda x: x
)

has_index_chain: Runnable = (
    RunnablePassthrough.assign(available_snippets_count=calculate_available_snippets_count)
    | RunnableBranch((lambda x: x["available_snippets_count"] > 0, retrieve_snippets_chain), lambda x: None)
    | RunnableLambda(lambda x: x.get("nodes") if x else [])
)

snippet_nodes_chain: Runnable = (
    # Context.setter("inputs")
    # | Context.setter("index", fetch_index)
    RunnablePassthrough.assign(index=fetch_index)
    | RunnableBranch((lambda x: x["index"], has_index_chain), lambda x: [])
).with_types(
    input_type=RetrieveFactsInputType  # type: ignore
)  # return list[NodeWithScore]

snippets_chain: Runnable = snippet_nodes_chain | RunnableLambda(lambda x: [i.get_text() for i in x])

# Adding callbacks to the chain enables automatic logging when the chain is invoked
chain = snippets_chain.with_config(callbacks=[LogFileCallbackHandler(logger=rag_logger)])  # type: ignore
