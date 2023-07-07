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
    """
    Ask the HeavyDB to establish a new db connection and return it's corresponding session id.

    Note: Don't call this command again and again. Since the session id will expire after an hour,
    you can re-use the previously generated session id's on endpoint calls until it gets timed-out.
    """
    heavydb_sessionid = HeavyDB.create_session_id(db_name=db_name)
    click.echo(heavydb_sessionid, color=True)
