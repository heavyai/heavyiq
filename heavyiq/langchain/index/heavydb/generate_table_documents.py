import os
import sys

sys.path.append("/Users/avinash/Heavy/heavynl")
from aiofiles.os import listdir as alistdir
import asyncio
from fastapi.concurrency import run_in_threadpool

from langchain.docstore.document import Document
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from heavyiq.config import get_config
from heavyiq.utils import awrite_to_file
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.llms import get_llm
from heavyiq.logging_utils import get_heavyiq_logger


table_summary_prompt = """With respect to the SQL table schema and sample data provided,
please create a comprehensive response encompassing the following aspects:

Description: Write a brief yet informative summary of the table's purpose and primary and secondary functionality.
Applications: Discuss potential use cases and applications.
Keywords: Identify a set of essential keywords and relationships that facilitate search, retrieval, and document indexing. Do not include column names directly.

Upon completion, evaluate the coherence, accuracy, and relevance of the generated response to ensure that it adheres to the requirements outlined above,
maximizing its value and usefulness for search and retrieval tasks."""


column_description_prompt = """Provided a table schema, sample rows, and common column values, please return an unnumbered list of each column with comments describing the column's purpose.
Do not return sample rows or common values. Do not explain any clauses. Do not include the column names which do not exists in the table schema.
Try to provide description for all the columns exsists on the table schema.

Example:
- table_name.column_name: Description of the column."""


async def aget_table_summary_document(heavydb: HeavyDB, table: str) -> Document:
    """
    Generates a document containing a summary of the specified table async.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the summary is to be generated.

    Returns:
        Document: A Document object containing the table summary and metadata.
    """
    table_info = await heavydb.aget_table_info([table])
    llm = await run_in_threadpool(get_llm, tags=["metadata_index", "table_summary"], temperature=0.2)
    messages = [
        SystemMessage(content=table_summary_prompt),
        HumanMessage(content=table_info),
    ]
    resp = await llm.apredict_messages(messages)
    return Document(page_content=resp.content, metadata={"source": table})


async def aget_table_column_description_document(heavydb: HeavyDB, table: str) -> Document:
    """
    Generates a document async containing the description of each column in a specified table.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the column descriptions are to be generated.

    Returns:
        Document: A Document object containing the column descriptions and metadata.
    """
    table_info = await heavydb.aget_table_info([table])
    llm = await run_in_threadpool(get_llm, tags=["metadata_index", "table_columns_description"], temperature=0.2)
    assert isinstance(llm, ChatOpenAI), "Custom llms for generating table_column_description document is not supported"
    # final llm should be a chat llm, so this function won't support custom llms
    messages = [
        SystemMessage(content=column_description_prompt),
        HumanMessage(content=table_info),
    ]
    resp = await llm.apredict_messages(messages)
    page_content = f"Column descriptions for {table} table:\n{resp.content}"
    return Document(page_content=page_content, metadata={"source": table})


async def acreate_and_write_table_document(heavydb: HeavyDB, table: str) -> None:
    """
    Creates and writes a document containing the table summary and column descriptions for a specified table asynchronously.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the document is to be created and written.
    """
    logger = get_heavyiq_logger()
    logger.info(f"Summarizing {table}...")
    docs = [
        aget_table_summary_document(heavydb, table),
        aget_table_column_description_document(heavydb, table),
    ]
    docs = await asyncio.gather(*docs)
    file_content = "\n\n".join([doc.page_content for doc in docs])
    file_path = f"table_documents/{table}.txt"
    await awrite_to_file(file_path, file_content)
    logger.info(f"Done writing table document {file_path}")


async def get_table_names_from_documents(folder_path: str) -> list[str]:
    table_names = []

    for file_name in await alistdir(folder_path):
        if file_name.endswith(".txt"):
            table_name = os.path.splitext(file_name)[0]
            table_names.append(table_name)

    return table_names


async def agenerate_table_documents() -> list[str]:
    """
    Generates and writes documents for all usable tables in the HeavyDB instance, containing table summaries and column descriptions async.
    Saves them to the table_documents directory.
    Returns a list of tables that had summaries generated.
    """
    CONFIG = get_config()
    logger = get_heavyiq_logger()
    tables_with_documents_already = await get_table_names_from_documents(CONFIG.table_documents_dir)
    logger.info("Generating summaries and column descriptions.")
    logger.debug(f"Tables with documents already (skipping): {tables_with_documents_already}")

    heavydb = await HeavyDB.from_env_async(ignore_tables=tables_with_documents_already)
    table_names_to_generate = heavydb.get_usable_table_names()

    tasks = []
    for table_name in table_names_to_generate:
        tasks.append(asyncio.create_task(acreate_and_write_table_document(heavydb=heavydb, table=table_name)))

    await asyncio.gather(*tasks)

    logger.info(
        f"Finished generating summaries and column descriptions for {len(list(table_names_to_generate))} tables."
    )

    return list(table_names_to_generate)


if __name__ == "__main__":
    asyncio.run(agenerate_table_documents())
