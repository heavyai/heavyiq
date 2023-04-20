import click


@click.group()
def metadata_index():
    """Metadata index management commands."""
    pass


@metadata_index.command()
@click.argument("table_name", type=str)
@click.pass_context
def generate_table_document(ctx: click.Context, table_name: str) -> None:
    """Generate a table document for a specified table. If file exists, it will be overwritten."""
    # Add the implementation for generating table document here
    click.echo(f"Generating table document for table: {table_name}")


@metadata_index.command()
@click.argument("table_name", type=str)
@click.pass_context
def reload_table_document(ctx: click.Context, table_name: str) -> None:
    """Reload the table document in index."""
    # Add the implementation for reloading table document here
    click.echo(f"Reloading table document for table: {table_name}")
