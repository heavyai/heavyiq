from pathlib import PosixPath
from typing import Any, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
from llama_index.readers.smart_pdf_loader import SmartPDFLoader

from heavyiq.langchain.heavydb import HeavyDB


class OverrideSmartPDFLoader(SmartPDFLoader):
    """
    Overrided smart pdf loader to return a big chunk of data w.r.t each section upon calling load_data method.
    """

    def load_data(self, pdf_path_or_url: str | PosixPath, extra_info: dict | None = None) -> list[Document]:
        """Load data and extract table from PDF file.

        Args:
            pdf_path_or_url (str): A url or file path pointing to the PDF

        Returns:
            List[Document]: List of documents.
        """
        pdf_path_or_url = str(pdf_path_or_url.resolve()) if isinstance(pdf_path_or_url, PosixPath) else pdf_path_or_url
        doc = self.pdf_reader.read_pdf(pdf_path_or_url)
        extra_info = extra_info or {}
        if "name" not in extra_info:
            extra_info["name"] = pdf_path_or_url.split("/")[-1]
        documents = []
        for section in doc.sections():
            # section info
            chunk = f"{section.to_context_text()}\n\n"
            # store child types involved (ie. para, table, list)
            chunk_types: list[str] = []
            for child in section.children:
                chunk += child.to_context_text(include_section_info=False)
                chunk_types.append(child.tag)
            document = Document(
                text=chunk,
                extra_info={**extra_info, "chunk_types": ",".join(set(chunk_types))},
            )
            documents.append(document)

        return documents


class TxtFileReader(BaseReader):
    """
    Class for reading .txt files.
    """

    def load_data(self, file: str, extra_info: dict | None = None) -> list[Document]:
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


class HeavyDBTableReader(BaseReader):
    """
    Supposed to read HeavyDB table.
    """

    def __init__(self, heavydb: HeavyDB, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.heavydb = heavydb

    async def aload_data(self, table_name: str, extra_info: dict | None = None) -> List[Document]:
        text = await self.heavydb.aget_single_table_info(
            table_name=table_name,
            include_samples=False,
            include_top_k=False,
            include_timestamp=False,
            include_comments=True,
        )
        extra_info = extra_info or {}
        extra_info["dbname"] = self.heavydb._dbname
        extra_info.update({"name": table_name, "type": "table"})
        return [Document(text=text, extra_info=extra_info)]
