from langchain_core.runnables import Runnable, RunnableLambda

from heavyiq.langchain.heavydb import get_config, get_db

CONFIG = get_config()


def _get_snippets_chain():
    """Lazy load snippets_chain to avoid ChromaDB initialization during module import."""
    from heavyrag.chains.retrieve_facts_chain import chain as snippets_chain
    return snippets_chain


async def _run_snippets(inputs: dict) -> dict:
    """Async runner to ensure coroutine chains are awaited properly."""
    chain = _get_snippets_chain()
    # Prefer async invocation to avoid TypeError when chain is coroutine-based
    return await chain.ainvoke(inputs)  # type: ignore


async def get_collection_name(inputs: dict) -> str:
    """
    Gets the heavydb name.
    """
    heavydb = await get_db(inputs["session_id"])
    return heavydb._dbname


async def form_relevant_info(infos: list) -> dict[str, list | str]:
    """
    Forms the relevant_info as a prompt patch.
    """
    node_ids = []
    node_contents = []
    for id_, text in infos:
        node_ids.append(id_)
        node_contents.append(text)
    joined_infos = "\n\n".join(node_contents)
    return {"snippet_ids": node_ids, "relevant_info": f"Relevant info:\n\n{joined_infos}\n" if joined_infos else ""}


chain: Runnable = (
    {
        "question": lambda x: x["question"],
        "collection_name": RunnableLambda(get_collection_name),  # type: ignore
        "similarity_cutoff": lambda x: CONFIG.rag_facts_similarity_cutoff_score,
        "similarity_top_k": lambda x: CONFIG.rag_facts_similarity_top_k,
        "use_reranker": lambda x: True,
        "reranker_top_k": lambda x: CONFIG.rag_facts_reranker_top_k,
        "reranker_cutoff": lambda x: CONFIG.rag_facts_reranker_cutoff_score,
    }
    | RunnableLambda(_run_snippets)  # type: ignore
    | form_relevant_info
)
# Sample Chain Output
# {"snippet_ids": ["fosioewwe23213", "dc3sddscdscdsc"], "relevant_info": f"Relevant info:\n\n{joined_infos}\n"}
# {"snippet_ids": [], "relevant_info": f"}
