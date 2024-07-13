from langchain_core.runnables import Runnable, RunnableLambda

from heavyiq.langchain.heavydb import get_config, get_db
from heavyrag.chains.retrieve_facts_chain import chain as snippets_chain

CONFIG = get_config()


async def get_collection_name(inputs: dict) -> str:
    """
    Gets the heavydb name.
    """
    heavydb = await get_db(inputs["session_id"])
    return heavydb._dbname


async def form_relevant_info(infos: list) -> str:
    """
    Forms the relevant_info as a prompt patch.
    """
    if not infos:
        return ""

    joined_infos = "\n\n".join(infos)
    return f"Relevant info:\n\n{joined_infos}\n"


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
    | snippets_chain
    | form_relevant_info
)
