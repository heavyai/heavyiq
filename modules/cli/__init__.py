import os

import click

from modules.config import config
from .agent_commands import agent
from .chain_commands import chain
from .metadata_index_commands import metadata_index


os.environ["OPENAI_API_KEY"] = config.openai_api_key


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


cli.add_command(agent)
cli.add_command(chain)
cli.add_command(metadata_index)
