# Module for loading documents using various readers such as PDFReader, DatabasReader, etc
from itertools import chain
from typing import Iterable

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode
from llama_index.readers.file import PDFReader

from heavyiq.config import get_config
from heavyiq.langchain.heavydb import HeavyDB
from heavyiq.utils import semaphore_gather
from heavyrag.readers import HeavyDBTableReader, OverrideSmartPDFLoader, TxtFileReader

CONFIG = get_config()


def get_meta(file_path: str) -> dict:
    """
    Helps to form metadata from the passed filepath.
    """
    splits = file_path.split("/")
    return {"file_path": file_path, "dbname": splits[-2], "name": splits[-1], "type": "document"}


def _exclude_metadata(documents: list[Document]) -> list[Document]:
    """
    Exclude metadata from documents.

    Args:
        documents (List[Document]): List of documents.
    """
    for doc in documents:
        doc.excluded_embed_metadata_keys.extend(
            [
                "file_path",
                "dbname",
                "name",
                "type",
            ]
        )
        doc.excluded_llm_metadata_keys.extend(
            [
                "file_path",
                "dbname",
                "name",
                "type",
            ]
        )

    return documents


async def aload_files_from_directory(source_dir: str) -> list[Document]:
    """
    Helps to load docs from a source_directory in recursive fashion.
    """

    reader = SimpleDirectoryReader(
        input_dir=source_dir,
        recursive=True,
        required_exts=[".pdf", ".txt"],
        file_extractor={
            ".txt": TxtFileReader(),
            # ".pdf": OverrideSmartPDFLoader(llmsherpa_api_url=CONFIG.rag_document_pdf_parser_url),
            ".pdf": PDFReader(),
        },
        file_metadata=get_meta,
    )
    documents = await reader.aload_data(show_progress=True)
    return _exclude_metadata(documents)


async def aload_file(filepath: str) -> list[Document]:
    """
    Create Documents from the passed file contents.
    """
    reader = None
    if filepath.endswith(".txt"):
        reader = TxtFileReader()
    elif filepath.endswith(".pdf"):
        reader = PDFReader()
    assert reader, "Invalid file to read!"
    documents = await reader.aload_data(filepath, extra_info=get_meta(filepath))
    return _exclude_metadata(documents)


async def aload_table(heavydb: HeavyDB, table_name: str) -> list[Document]:
    """
    Load HeavyDB table.
    """
    reader = HeavyDBTableReader(heavydb=heavydb)
    documents = await reader.aload_data(table_name=table_name)
    return _exclude_metadata(documents)


async def aload_specific_tables(heavydb: HeavyDB, tables: Iterable[str]) -> list[Document]:
    """
    Load specific tables.
    """
    reader = HeavyDBTableReader(heavydb=heavydb)
    tasks = [reader.aload_data(table_name=i) for i in tables]
    result = await semaphore_gather(5, tasks)  # run only 5 tasks at a time
    flattened_list = list(chain.from_iterable(result))
    return _exclude_metadata(flattened_list)


async def aload_tables(heavydb: HeavyDB, exlude_tables: list[str] | None = None) -> list[Document]:
    """
    Load all tables.
    """
    exlude_tables = exlude_tables or []
    reader = HeavyDBTableReader(heavydb=heavydb)
    tasks = [reader.aload_data(table_name=i) for i in heavydb.get_usable_table_names() if i not in exlude_tables]
    result = await semaphore_gather(5, tasks)  # run only 5 tasks at a time
    flattened_list = list(chain.from_iterable(result))
    return _exclude_metadata(flattened_list)


def load_facts(facts: list[tuple[str, str]], heavydb_name: str, table_name: str | None = None) -> list[Document]:
    """
    Form a document from the passed facts.
    """
    return [
        TextNode(text=fact, id_=fact_id, metadata={"dbname": heavydb_name, "type": "facts", "id": fact_id})
        for fact_id, fact in facts
    ]


def load_fact(fact_id: str, fact: str, heavydb_name: str) -> TextNode:
    """
    Form a textnode from fact string.
    """
    return TextNode(text=fact, id_=fact_id, metadata={"dbname": heavydb_name, "type": "facts", "id": fact_id})
