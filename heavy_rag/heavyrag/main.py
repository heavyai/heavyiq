import logging
import os

from heavyrag.ingest import get_vectorstore, reload_index
from heavyrag.read import HeavyDBReader
from heavyrag.utils import get_existing_tables_from_collection
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from heavyrag.settings import Settings

logger = logging.getLogger(__name__)
# dict mapping of dbname and bool to represent whether an index building is in progress for
# a particular database or not
INDEX_ON_PROGRESS: dict[str, bool] = {}


def get_heavydb_reader(
    sessionid: str, dbhost: str | None = None, dbport: int | None = None
) -> HeavyDBReader:
    """
    Gets HeavyDB reader object.
    """
    settings = Settings()
    reader = HeavyDBReader(
        host=settings.heavydb_host,
        port=settings.heavydb_port,
        protocol=settings.heavydb_protocol,
        sessionid=sessionid,
    )
    return reader


def get_or_create_index(
    sessionid: str | None = None, reader: HeavyDBReader | None = None
) -> VectorStoreIndex:
    """
    Get or create VectorStore Index.
    if not exists, create a new one by querying the heavydb for all the table schemas and then load it to the index.
    if exists, then load only the missing tables into the index.
    """
    settings = Settings()
    if not reader:
        if not sessionid:
            raise ValueError("sessionid needs to be passed!")
        reader = HeavyDBReader(
            host=settings.heavydb_host,
            port=settings.heavydb_port,
            protocol=settings.heavydb_protocol,
            sessionid=sessionid,
        )
    conn_dbname = reader.database_name
    if conn_dbname in INDEX_ON_PROGRESS:
        raise ValueError(
            f"Index building in-progress for {conn_dbname} database, please wait."
        )

    if not os.path.exists(settings.persistant_storage_dir):
        logger.info("Creating new index...")
        try:
            INDEX_ON_PROGRESS[conn_dbname] = True
            index = reload_index(reader, settings=settings)
        except Exception as e:
            raise e
        finally:
            INDEX_ON_PROGRESS.pop(conn_dbname, None)
        logger.info("Index created successfully.")
    else:
        logger.info("Index already exists, so loading it from the cache dir.")
        index = load_index_from_storage(
            StorageContext.from_defaults(
                persist_dir=settings.persistant_storage_dir,
                vector_store=get_vectorstore(
                    settings.persistant_collection_dir,
                    collection_name=settings.collection_name,
                ),
            ),
            embed_model=HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
            # show_progress=True
        )  # type: ignore
        # grab the chromadb collection and then search for existing tables
        existing_tables = get_existing_tables_from_collection(
            index._vector_store.client, conn_dbname
        )
        if sorted(reader.tables) != sorted(existing_tables):
            # reload index in-case of missing tables
            logger.info(
                "No tables or Some tables not found in the index, triggering a reload of the index."
            )
            try:
                INDEX_ON_PROGRESS[conn_dbname] = True
                index = reload_index(
                    reader, settings=settings, exclude_tables=existing_tables
                )
            except Exception as e:
                raise e
            finally:
                INDEX_ON_PROGRESS.pop(conn_dbname, None)
            logger.info("Index reloaded successfully.")

    return index


def retrieve_index(
    index: VectorStoreIndex, question: str, dbname: str | None = None
) -> dict[str, float]:
    """
    Retrieve nodes from the index based on the question.
    """
    # define metadata filters
    # if dbname is being passed then filter only on the nodes having a particular database metadata
    filters = (
        MetadataFilters(
            filters=[
                MetadataFilter(key="database", operator=FilterOperator.EQ, value=dbname)
            ]
        )
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
    index = get_or_create_index(reader=dbreader)
    tables_with_score = retrieve_index(index, question, dbname=dbreader.database_name)
    return list(tables_with_score.keys())[:n]
