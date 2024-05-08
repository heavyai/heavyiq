# Module for loading documents using various readers such as PDFReader, DatabasReader, etc
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from heavyiq.config import get_config
from heavyrag.readers import OverrideSmartPDFLoader, TxtFileReader

CONFIG = get_config()


def get_meta(file_path: str) -> dict:
    """
    Helps to form metadata from the passed filepath.
    """
    splits = file_path.split("/")
    return {"file_path": file_path, "dbname": splits[-2], "filename": splits[-1], "type": "document"}


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
                "filename",
                "type",
            ]
        )
        doc.excluded_llm_metadata_keys.extend(
            [
                "file_path",
                "dbname",
                "filename",
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
            ".pdf": OverrideSmartPDFLoader(llmsherpa_api_url=CONFIG.rag_document_pdf_parser_url),
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
        reader = OverrideSmartPDFLoader(llmsherpa_api_url=CONFIG.rag_document_pdf_parser_url)
    assert reader, "Invalid file to read!"
    documents = await reader.aload_data(filepath, extra_info=get_meta(filepath))
    return _exclude_metadata(documents)
