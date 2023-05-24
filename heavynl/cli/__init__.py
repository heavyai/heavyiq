import os

import click

from heavynl.config import get_config
from .agent_commands import agent
from .chain_commands import chain
from .metadata_index_commands import metadata_index


os.environ["OPENAI_API_KEY"] = get_config().openai_api_key


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


cli.add_command(agent)
cli.add_command(chain)
cli.add_command(metadata_index)
