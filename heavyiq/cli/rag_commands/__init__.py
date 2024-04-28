import asyncio
import pathlib
from collections import defaultdict

import click
from heavyrag.documents.ingest import sync_doc_index
from heavyrag.documents.read import DocumentDatabaseReader
from heavyrag.settings import settings

from heavyiq.cli.decorators import coro
from heavyiq.cli.eval_commands import eval
from heavyiq.langchain import HeavyDB
from heavyiq.rag.document.database import Session as DBSession
from heavyiq.rag.document.database import engine
from heavyiq.rag.document.operations import insert_document


@click.group()
def rag():
    """RAG commands"""
    pass


def list_documents_grouped_by_heavydb_name(parent_dir: pathlib.Path) -> dict:
    """
    Return a list of documents path grouped by heavydb database names.
    Example:
    {
        "heavyiq": ["/abs/path/to/heavyiq.pdf", "/path/to/heavyiq_installation.pdf"],
        "heavyai": ["/path/to/heavydb.pdf"]
    }
    """
    documents = defaultdict(list)
    # Check if the parent directory exists
    if parent_dir.exists() and parent_dir.is_dir():
        # Iterate over directories inside the parent directory
        for database_directory in parent_dir.iterdir():
            if database_directory.is_dir():
                for fil in database_directory.iterdir():
                    if fil.is_file():
                        documents[database_directory.name].append(str(fil.resolve()))
    else:
        raise ValueError(f"The parent directory {parent_dir} does not exist or is not a directory.")

    return documents


async def import_documents(heavydb_name: str, files: list[str]) -> bool:
    """
    Import documents.
    """
    try:
        session = DBSession()
        tasks = [insert_document(session, heavydb_name, file_path=f) for f in files]
        await asyncio.gather(*tasks)
        reader = DocumentDatabaseReader(engine=session.bind)
        sync_doc_index(reader, heavydb_name)
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"Failed to import files for database {heavydb_name}, {e}")
        return False
    return True


@rag.command()
@coro
@click.argument("rag_documents_folder", type=str)
@click.pass_context  # type: ignore
async def run_documents_import(ctx: click.Context, rag_documents_folder: str) -> None:
    """
    Eval command for uploading documents for RAG processing.
    """
    parent_folder = pathlib.Path(rag_documents_folder)
    tasks = []
    for heavydb_name, files in list_documents_grouped_by_heavydb_name(parent_folder).items():
        print(heavydb_name, files)
        heavydb = await HeavyDB.from_env_async(db_name=heavydb_name)
        tasks.append(import_documents(heavydb._dbname, files))

    await asyncio.gather(*tasks)
    print(f"Successfully imported documents from {rag_documents_folder} folder.")


@rag.command()
@coro
@click.option("--file", default="", help="Specific file", type=str)
@click.option(
    "--llm-model",
    default="heavyai/heavyiq-deepseek-coder-7b-combo-v60-3072-tokens-questions-no-meta-lr-1-7-gc-1-5",
    help="VLLM model",
    type=str,
)
@click.option(
    "--llm-url", default="http://209.20.159.184:5000/v1", help="VLLM server url which ends with /v1", type=str
)
@click.argument("question", type=str)
@click.argument("dbname", type=str)
@click.pass_context  # type: ignore
async def ask(ctx: click.Context, question: str, dbname: str, file: str, llm_url: str, llm_model: str):
    """
    Ask questions about the uploaded documents specific to a particular database
    or specific to a particular file.
    """
    settings.llm_api_url = llm_url
    settings.llm_model = llm_model
    from heavyrag.documents.etl import DocumentIndexETL

    etl = DocumentIndexETL(
        collection_name=dbname,
        reader_init_kwargs={"engine": engine},
    )
    response = etl.run(question)
    print(response.response)
