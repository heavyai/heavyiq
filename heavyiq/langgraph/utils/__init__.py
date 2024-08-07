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
        yield ctx
    except KeyError:
        # raise ValueError('HeavyDB Context expects "session_id" to be passed as part of the config.')
        yield None
    finally:
        heavydb_var.set(None)
        if ctx:
            ctx.db._conn.close()
