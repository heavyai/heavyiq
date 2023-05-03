from pathlib import Path

import click
from langchain.vectorstores import Chroma

from heavynl.langchain import HeavyDB
from heavynl.langchain.index import (
    create_and_write_table_document,
    get_vectorstore_index_creator,
    update_tables_in_index,
)
from heavynl.config import config


@click.group()
def metadata_index():
    """Metadata index management commands."""
    pass


@metadata_index.command()
@click.argument("table_name", type=str)
@click.pass_context
def generate_table_document(ctx: click.Context, table_name: str) -> None:
    """Generate a table document for a specified table. If file exists, it will be overwritten."""
    heavydb = HeavyDB.from_env(include_tables=[table_name])
    click.echo(f"Generating table document for table: {table_name}")
    create_and_write_table_document(heavydb, table_name)
    click.echo(f"Done. You likely want to run `metadata-index reload-table-document {table_name}` now.")


@metadata_index.command()
@click.argument("table_name", type=str)
@click.pass_context
def reindex_table_document(ctx: click.Context, table_name: str) -> None:
    """Reload the table document in index."""
    # Add the implementation for reloading table document here
    persist_path = Path(config.metadata_index_dir)
    if not persist_path.exists():
        raise click.ClickException(f"Index does not exist at configured path: {persist_path}")
    click.echo(f"Reloading table document for table: {table_name}")
    index_creator = get_vectorstore_index_creator(config.metadata_index_dir)
    vectorstore = Chroma(embedding_function=index_creator.embedding, persist_directory=config.metadata_index_dir)
    update_tables_in_index(index_creator, vectorstore, [table_name])
    vectorstore.persist()
