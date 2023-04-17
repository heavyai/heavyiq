from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator

from modules.config import config
from modules.langchain.index import generate_table_documents
from modules.langchain.index.heavydb_metadata_index import HeavyDBMetadataIndex

huggingface_model_name = config.huggingface_embed_model


def read_table_documents(folder_path: str, include: Optional[list[str]] = None) -> Iterator[Document]:
    folder = Path(folder_path)
    for file_name in folder.iterdir():
        if file_name.suffix == ".txt":
            table_name = file_name.stem
            if include is not None and table_name not in include:
                continue
            with open(file_name, "r") as f:
                file_contents = f.read()
            yield Document(page_content=file_contents, metadata={"source": table_name})


def update_existing_index(
    index_creator: VectorstoreIndexCreator, vectorstore: Chroma, folder_path: str, tables_to_update: list[str]
):
    print("New table summaries will replace existing summaries in the index.")
    if len(tables_to_update) == 1:
        vectorstore._collection.delete(where={"source": tables_to_update[0]})
    else:
        vectorstore._collection.delete(where={"$or": [{"source": table_name} for table_name in tables_to_update]})
    print("Reading new table documents...")
    docs = list(read_table_documents(folder_path, include=tables_to_update))
    print("Splitting documents...")
    sub_docs = index_creator.text_splitter.split_documents(docs)
    print(f"Indexing documents with {huggingface_model_name}...")
    vectorstore.add_documents(sub_docs)
    print("Done")


def create_index_if_nonexistent(
    folder_path: str = "table_documents", persist_directory: str = "db"
) -> HeavyDBMetadataIndex:
    """
    Create a new vector store index if it does not exist, otherwise return the existing index.

    :param folder_path: Path to the folder containing table document files.
    :param persist_directory: Path to the directory where the vector store index should be persisted.
    :return: A HeavyDBMetadataIndex instance containing the vector store index.
    """
    tables_with_new_summaries = generate_table_documents(folder_path)

    index_creator = VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": persist_directory},
        embedding=HuggingFaceEmbeddings(model_name=huggingface_model_name),
    )

    persist_path = Path(persist_directory)
    if persist_path.exists():
        print("Index already exists. Returning existing index.")
        vectorstore: Chroma = Chroma(embedding_function=index_creator.embedding, persist_directory=persist_directory)
        if len(tables_with_new_summaries) > 0:
            update_existing_index(index_creator, vectorstore, folder_path, tables_with_new_summaries)
    else:
        print("Index does not exist. Creating new index.")
        print("Reading table documents...")
        docs = read_table_documents(folder_path)

        print("Splitting documents...")
        sub_docs = index_creator.text_splitter.split_documents(list(docs))
        print(f"Indexing documents with {huggingface_model_name}...")
        vectorstore = index_creator.vectorstore_cls.from_documents(
            sub_docs, index_creator.embedding, **index_creator.vectorstore_kwargs
        )
        print("Done")

    return HeavyDBMetadataIndex(vectorstore=vectorstore)


heavydb_index = create_index_if_nonexistent()
