from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig

from heavyiq.langchain.heavydb import HeavyDB, heavydb_var


class HeavyDBContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    db: HeavyDB


@asynccontextmanager
async def make_heavydb_context(config: RunnableConfig) -> AsyncGenerator[HeavyDBContext, None]:
    """
    Context manager responsible for serving HeavyDBContext instance.
    """
    # here you could read the config values passed invoke/stream to customize the context object

    # as an example, we create an heavydb connection, which could then be used in your graph's nodes
    ctx: HeavyDBContext | None = None
    try:
        session_id = config["configurable"]["session_id"]
        db = await HeavyDB.from_session_async(session_id=session_id)
        heavydb_var.set(db)
        ctx = HeavyDBContext(db=db)
        yield None
    except KeyError:
        # raise ValueError('HeavyDB Context expects "session_id" to be passed as part of the config.')
        yield None
    finally:
        heavydb_var.set(None)
        if ctx:
            ctx.db._conn.close()


# reducer
def remove_duplicates(left: list | None, right: list | None) -> list:
    """
    Reducer function that removes duplicate items from a list.

    Args:
        left (list | None): The existing list of items.
        right (list | None): The new list of items to be added.

    Returns:
        list: The updated list with duplicates removed.
    """
    if left is None:
        left = []
    if right is None:
        right = []

    # Create a set to store unique items
    unique_items = set(left)

    # Add new items to the set, which will automatically remove duplicates
    unique_items.update(right)

    # Convert the set back to a list and return it
    return list(unique_items)
