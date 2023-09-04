from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .overrides import OverrideGitbookLoader


def get_latest_heavyai_docs() -> list[Document]:
    docs_loader = OverrideGitbookLoader(
        "https://docs.heavy.ai",
        load_all_paths=True,
        exclude_paths=["/v/"],
        include_paths=["/immerse/", "/sql/", "/heavyconnect/", "/heavyrf/", "/heavyml/"],
    )
    all_pages_data = docs_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(all_pages_data)
    return documents
