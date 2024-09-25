import click

from heavyiq.langchain.utils import init_telemetrics

# from .chain_commands import chain
from .eval_commands import eval
from .gen_commands import gen
from .heavydb_commands import db
from .rag_commands import rag

init_telemetrics()


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    ctx.ensure_object(dict)


# cli.add_command(chain)
cli.add_command(eval)
cli.add_command(gen)
cli.add_command(db)
cli.add_command(rag)
