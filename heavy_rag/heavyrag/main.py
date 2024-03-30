import logging
import os

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from heavyrag.ingest import sync_index
from heavyrag.read import HeavyDBReader
from heavyrag.settings import settings
from heavyrag.utils import get_existing_tables_from_collection

logger = logging.getLogger(__name__)
# dict mapping of dbname and bool to represent whether an index building is in progress for
# a particular database or not
INDEX_ON_PROGRESS: dict[str, bool] = {}


def get_heavydb_reader(sessionid: str, dbhost: str | None = None, dbport: int | None = None) -> HeavyDBReader:
    """
    Gets HeavyDB reader object.
    """
    reader = HeavyDBReader(
        host=dbhost or settings.heavydb_host,
        port=dbport or settings.heavydb_port,
        protocol=settings.heavydb_protocol,
        sessionid=sessionid,
    )
    return reader


def get_or_sync_index(reader: HeavyDBReader) -> VectorStoreIndex:
    """
    Get or create or update VectorStore Index.
    If not exists, create a new one by querying the heavydb for all the table schemas and then load it into the index.
    If exists, then load only the missing tables into the index.
    """
    conn_dbname = reader.database_name
    if conn_dbname in INDEX_ON_PROGRESS:
        raise ValueError(f"Index building in-progress for {conn_dbname} database, please wait.")

    try:
        INDEX_ON_PROGRESS[conn_dbname] = True
        index = sync_index(reader)
    except Exception as e:
        raise e
    finally:
        INDEX_ON_PROGRESS.pop(conn_dbname, None)
    logger.info("Index synced successfully.")
    return index


def retrieve_index(index: VectorStoreIndex, question: str, dbname: str | None = None) -> dict[str, float]:
    """
    Retrieve nodes from the index based on the question.
    """
    # define metadata filters
    # if dbname is being passed then filter only on the nodes having a particular database metadata
    filters = (
        MetadataFilters(filters=[MetadataFilter(key="database", operator=FilterOperator.EQ, value=dbname)])
        if dbname
        else None
    )
    retriever = index.as_retriever(
        similarity_top_k=10,
        filters=filters,
    )
    out: list[NodeWithScore] = retriever.retrieve(question)
    seen: dict[str, float] = {}
    for i in out:
        table = i.node.metadata["table"]
        if table not in seen:
            seen[table] = i.score  # type: ignore
    # returns table_name, score mapping dict
    return seen


def ask(sessionid: str, question: str, n: int = 2) -> list[str]:
    """
    Ask a question against the stored index.
    Returns a list of possible tables which are likely to be associated with the asked question.

    Args:
        sessionid (str): HeavyDB session id.
        question (str) : Natural Language Question.
              n (int)  : Number of tables to return.

    Returns:
        A list of matched table names.
    """
    dbreader = get_heavydb_reader(sessionid)
    index = get_or_sync_index(reader=dbreader)
    tables_with_score = retrieve_index(index, question, dbname=dbreader.database_name)
    return list(tables_with_score.keys())[:n]
