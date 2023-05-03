from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from langchain.docstore.document import Document

from heavynl.config import config

table_documents_dir = config.table_documents_dir


def read_table_documents(include: Optional[list[str]] = None) -> Iterator[Document]:
    """
    Iterate over a folder containing table documents and yield Document instances.
    :param include: Optional list of table names to include. If None, all tables will be included.
    :return: An iterator yielding Document instances for each table document.
    """
    folder = Path(table_documents_dir)
    for file_name in folder.iterdir():
        if file_name.suffix == ".txt":
            table_name = file_name.stem
            if include is not None and table_name not in include:
                continue
            with open(file_name, "r") as f:
                file_contents = f.read()
            yield Document(page_content=file_contents, metadata={"source": table_name})


def apply_retriever_filter(search_kwargs: dict, allowable_tables: Optional[list[str]] = None) -> dict:
    """Apply a filter to the search_kwargs to only search for tables in the allowable_tables list.
    If allowable_tables is None, no filter is applied."""
    if allowable_tables is not None:
        if len(allowable_tables) == 0:
            raise Exception("No tables permitted to be searched.")
        if len(allowable_tables) == 1:
            search_kwargs["filter"] = {"source": allowable_tables[0]}
        else:
            search_kwargs["filter"] = {"$or": [{"source": table_name} for table_name in allowable_tables]}
    return search_kwargs
