from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores.base import VectorStore

from heavyiq.config import get_config

from ..utils import get_vectorstore_index_creator
from .heavyai_docs_index import HeavyAIDocsIndex
from .utils import get_latest_heavyai_docs

heavyai_docs_index = None


def create_heavyai_docs_index() -> HeavyAIDocsIndex:
    global heavyai_docs_index
    if heavyai_docs_index is not None:
        return heavyai_docs_index
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
        documents = get_latest_heavyai_docs()
        vectorstore = index_creator.vectorstore_cls.from_documents(
            documents, index_creator.embedding, **index_creator.vectorstore_kwargs
        )

    heavyai_docs_index = HeavyAIDocsIndex(vectorstore=vectorstore, text_splitter=index_creator.text_splitter)  # type: ignore
    return heavyai_docs_index
