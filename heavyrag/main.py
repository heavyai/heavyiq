import asyncio
from collections import defaultdict

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.indices.base import BaseIndex
from llama_index.core.response_synthesizers import ResponseMode

from heavyrag.evaluate import aevaluate_response_by_relevancy
from heavyrag.filters import get_document_filter_matches
from heavyrag.index import acreate_index_and_insert_nodes, get_index
from heavyrag.llm import get_llm
from heavyrag.loaders import aload_file, aload_files_from_directory
from heavyrag.prompts import TEXT_QA_PROMPT
from heavyrag.transform import atransform


async def ainsert_documents(source_dir: str) -> list[BaseIndex]:
    """
    Grab documents from a source folder recursivley, split, transfor ingest into vectordb, finally create index.
    """
    documents = await aload_files_from_directory(source_dir)
    nodes = await atransform(documents=documents)
    # group the nodes by database id
    nodes_group_by_database_name = defaultdict(list)
    for node in nodes:
        try:
            nodes_group_by_database_name[node.metadata["dbname"]].append(node)
        except KeyError:
            print(node.get_content())
            pass

    tasks = []
    for dbname in nodes_group_by_database_name:
        db_nodes = nodes_group_by_database_name[dbname]
        tasks.append(
            acreate_index_and_insert_nodes(db_nodes, collection_name=dbname, collection_metadata={"type": "document"})
        )

    return await asyncio.gather(*tasks)


async def ainsert_document(filepath: str, heavydb_name: str) -> BaseIndex:
    """
    Inserts a document into an existing or newly created index.
    """
    documents = await aload_file(filepath)
    nodes = await atransform(documents=documents)
    index = get_index(collection_name=heavydb_name)
    if not index:
        # index not found so create a new one
        index = await acreate_index_and_insert_nodes(nodes=nodes, collection_name=heavydb_name)
    else:
        # index already exists, so just add nodes to it
        index.insert_nodes(nodes=nodes, show_progress=True)

    return index


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
