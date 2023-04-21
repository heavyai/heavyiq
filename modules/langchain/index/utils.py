from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from langchain.docstore.document import Document

from modules.config import config

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
