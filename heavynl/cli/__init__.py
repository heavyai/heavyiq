import click

from .agent_commands import agent
from .chain_commands import chain
from .eval_commands import eval
from .metadata_index_commands import metadata_index
from .heavydb_commands import db


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


cli.add_command(agent)
cli.add_command(chain)
cli.add_command(eval)
cli.add_command(metadata_index)
cli.add_command(db)
