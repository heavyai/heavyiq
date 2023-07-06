import click
from heavynl.langchain import HeavyDB


@click.group()
def db():
    """HeavyDB commands"""
    pass


@db.command()
@click.option("--db-name", default="", help="HeavyDB database to connect with.", type=str)
@click.pass_context
def session(ctx: click.Context, db_name: str) -> None:
    """Ask the HeavyDB to create and return the session id."""
    heavydb_sessionid = HeavyDB.create_session_id(db_name=db_name)
    click.echo(heavydb_sessionid, color=True)
