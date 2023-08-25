from pathlib import Path

from langchain.document_loaders import GitbookLoader
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore

from heavyiq.config import get_config

from .heavyai_docs_index import HeavyAIDocsIndex
from ..utils import get_vectorstore_index_creator


def create_heavyai_docs_index() -> HeavyAIDocsIndex:
    config = get_config()
    docs_index_dir = config.docs_index_dir

    index_creator = get_vectorstore_index_creator(docs_index_dir)

    persist_path = Path(docs_index_dir)
    vectorstore: VectorStore
    if persist_path.exists():
        print("Docs Index already exists. Returning existing index.")
        vectorstore = Chroma(embedding_function=index_creator.embedding, persist_directory=docs_index_dir)
    else:
        print("Docs Index does not exist. Creating new index.")
        docs_loader = GitbookLoader("https://docs.heavy.ai", load_all_paths=True)
        all_pages_data = docs_loader.load()  # this loads all docs, even old versions
        # filter out all old versions. older versions can be determined if doc.metadata.source includes '/v/'
        all_pages_data = [page_data for page_data in all_pages_data if "/v/" not in page_data.metadata.source]  # type: ignore
        for doc in all_pages_data:
            print(doc)

    return None
