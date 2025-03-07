from pathlib import Path

import click
from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma

from heavyiq.utils import is_path_exists
from heavyiq.cli.decorators import coro
from heavyiq.langchain import HeavyDB
from heavyiq.langchain.index.heavydb import (
    create_and_write_table_document,
    acreate_and_write_table_document,
    get_vectorstore_index_creator,
    update_tables_in_index,
    aupdate_tables_in_index,
)
from heavyiq.config import get_config


@click.group()
def metadata_index():
    """Metadata index management commands."""
    pass


@metadata_index.command()
@coro
@click.argument("table_name", type=str)
@click.pass_context  # type: ignore
async def generate_table_document(ctx: click.Context, table_name: str) -> None:
    """Generate a table document for a specified table. If file exists, it will be overwritten."""
    heavydb = await HeavyDB.from_env_async(include_tables=[table_name])
    click.echo(f"Generating table document for table: {table_name}")
    await acreate_and_write_table_document(heavydb, table_name)
    click.echo(f"Done. You likely want to run `metadata-index reload-table-document {table_name}` now.")


@metadata_index.command()
@coro
@click.argument("table_name", type=str)
@click.pass_context  # type: ignore
async def reindex_table_document(ctx: click.Context, table_name: str) -> None:
    """Reload the table document in index."""
    # Add the implementation for reloading table document here
    config = get_config()
    if not await is_path_exists(config.metadata_index_dir):
        raise click.ClickException(f"Index does not exist at configured path: {config.metadata_index_dir}")
    click.echo(f"Reloading table document for table: {table_name}")
    index_creator = get_vectorstore_index_creator(config.metadata_index_dir)
    vectorstore = Chroma(embedding_function=index_creator.embedding, persist_directory=config.metadata_index_dir)
    await aupdate_tables_in_index(index_creator, vectorstore, [table_name])
    await run_in_threadpool(vectorstore.persist)
