from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

from langchain.docstore.document import Document
from langchain.schema import HumanMessage, SystemMessage

from heavynl.config import get_config
from heavynl.langchain import HeavyDB
from heavynl.langchain.llms import get_chat_llm


table_summary_prompt = """With respect to the SQL table schema and sample data provided,
please create a comprehensive response encompassing the following aspects:

Description: Write a brief yet informative summary of the table's purpose and primary and secondary functionality.
Applications: Discuss potential use cases and applications.
Keywords: Identify a set of essential keywords and relationships that facilitate search, retrieval, and document indexing. Do not include column names directly.

Upon completion, evaluate the coherence, accuracy, and relevance of the generated response to ensure that it adheres to the requirements outlined above,
maximizing its value and usefulness for search and retrieval tasks."""

CONFIG = get_config()


def get_table_summary_document(heavydb: HeavyDB, table: str) -> Document:
    """
    Generates a document containing a summary of the specified table.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the summary is to be generated.

    Returns:
        Document: A Document object containing the table summary and metadata.
    """
    table_info = heavydb.get_table_info([table])
    llm = get_chat_llm(["metadata_index", "table_summary"], model=CONFIG.openai_gpt_model, temperature=0.2)
    messages = [
        SystemMessage(content=table_summary_prompt),
        HumanMessage(content=table_info),
    ]
    resp = llm(messages)
    print(resp.content)
    return Document(page_content=resp.content, metadata={"source": table})


column_description_prompt = """Provided a table schema, sample rows, and common column values, please return an unnumbered list of each column with comments describing the column's purpose.
Do not return sample rows or common values. Do not explain any clauses.

Example:
- table_name.column_name: Description of the column."""


def get_table_column_description_document(heavydb: HeavyDB, table: str) -> Document:
    """
    Generates a document containing the description of each column in a specified table.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the column descriptions are to be generated.

    Returns:
        Document: A Document object containing the column descriptions and metadata.
    """
    table_info = heavydb.get_table_info([table])
    llm = get_chat_llm(["metadata_index", "table_columns_description"], model=CONFIG.openai_gpt_model, temperature=0.2)
    messages = [
        SystemMessage(content=column_description_prompt),
        HumanMessage(content=table_info),
    ]
    resp = llm(messages)
    page_content = f"Column descriptions for {table} table:\n{resp.content}"
    return Document(page_content=page_content, metadata={"source": table})


def create_and_write_table_document(heavydb: HeavyDB, table: str) -> None:
    """
    Creates and writes a document containing the table summary and column descriptions for a specified table.

    Args:
        heavydb (HeavyDB): An instance of HeavyDB containing the table information.
        table (str): The name of the table for which the document is to be created and written.
    """
    print(f"Summarizing {table}...")
    docs = [
        get_table_summary_document(heavydb, table),
        get_table_column_description_document(heavydb, table),
    ]
    file_content = ""
    for doc in docs:
        file_content += doc.page_content + "\n\n"
        print(doc.page_content)
    write_to_file(f"table_documents/{table}.txt", file_content)


def write_to_file(file_name: str, content: str):
    """
    Writes the given content to a specified file.

    Args:
        file_name (str): The name of the file to write the content to.
        content (str): The content to be written to the file.
    """
    with open(file_name, "w") as f:
        f.write(content)


def get_table_names_from_documents(folder_path: str) -> list[str]:
    table_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            table_name = os.path.splitext(file_name)[0]
            table_names.append(table_name)

    return table_names


def generate_table_documents() -> list[str]:
    """
    Generates and writes documents for all usable tables in the HeavyDB instance, containing table summaries and column descriptions.
    Saves them to the table_documents directory.
    Returns a list of tables that had summaries generated.
    """
    tables_with_documents_already = get_table_names_from_documents(CONFIG.table_documents_dir)
    print("Generating summaries and column descriptions.")
    print(f"Tables with documents already (skipping): {tables_with_documents_already}")
    heavydb = HeavyDB.from_env(ignore_tables=tables_with_documents_already)
    process_func = partial(create_and_write_table_document, heavydb)
    table_names_to_generate = heavydb.get_usable_table_names()
    with ThreadPoolExecutor() as executor:
        executor.map(process_func, table_names_to_generate)
    print(f"Finished generating summaries and column descriptions for {len(list(table_names_to_generate))} tables.")

    return list(table_names_to_generate)


if __name__ == "__main__":
    generate_table_documents()
