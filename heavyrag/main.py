from llama_index.core import get_response_synthesizer
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import MetadataMode

from heavyiq.langchain.heavydb import HeavyDB
from heavyrag.evaluate import aevaluate_response_by_relevancy
from heavyrag.filters import get_document_filter_matches, table_filters
from heavyrag.index import get_index
from heavyrag.ingest import sync_table_index
from heavyrag.llm import get_llm
from heavyrag.prompts import REFINE_TABLE_NAME_PROMPT, TEXT_QA_PROMPT


async def ask_document(
    question: str, heavydb_name: str, file_name: str | None = None, do_evaluate: bool = True
) -> tuple[RESPONSE_TYPE, EvaluationResult | None]:
    """
    Ask questions about a document or a group of documents mapped to a particular database.
    """
    index = get_index(collection_name=heavydb_name)
    if not index:
        raise ValueError(f"VectorStoreIndex not found {heavydb_name} collection.")

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
        # n.metadata.update({**n.metadata, "table_name": n.metadata["name"]})
        table_name = n.metadata["name"]
        if table_name not in tables:
            tables.append(table_name)

    # response_synthesizer = get_response_synthesizer(
    #     llm=get_llm(), response_mode=ResponseMode.SIMPLE_SUMMARIZE, text_qa_template=REFINE_TABLE_NAME_PROMPT
    # )
    # print(response_synthesizer.get_prompts())
    # response = response_synthesizer.synthesize(question, nodes=nodes_with_score, existing_answer=",".join(tables))

    # return response.response
    return tables
