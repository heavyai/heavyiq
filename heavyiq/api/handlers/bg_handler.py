import asyncio
from enum import Enum, auto

from fastapi import BackgroundTasks, Request
from fastapi.concurrency import run_in_threadpool

from heavyiq.api.models import UpdateIndexRequest, UpdateIndexResponse
from heavyiq.langchain.index.heavydb import acreate_index_if_nonexistent, create_index_if_nonexistent
from heavyiq.logging_utils import _get_access_logger, get_heavyiq_logger
from heavyiq.utils import SharedDictSingleton


class UpdateIndexStatus(Enum):
    """Update Index status enum"""

    started = auto()
    in_progress = auto()


async def my_task(dict_: SharedDictSingleton, key: str):
    print("on my task")
    await dict_.put(key, True)
    await asyncio.sleep(100)
    await dict_.put(key, False)
    print("finished my task")


async def background_task():
    print("I run on the background")
    await asyncio.sleep(500)  # Use asyncio.sleep instead of time.sleep
    print("Finished run on bg.")


async def start_update_index_background_task(shared_dict: SharedDictSingleton, key: str, session: str | None = None):
    """
    Triggers the table document generation and chromadb index update on the background.

    Args:
        session: heavydb
        shared_dict: Shared dict which is supposed to shared among forked processes
        key: dict key
    """
    logger = get_heavyiq_logger()
    await shared_dict.put(key, True)
    try:
        # sleep is necessary here, which pauses the current task for specific time, allowing other task to run in the meantime
        await asyncio.sleep(20)
        await acreate_index_if_nonexistent(session=session)
        logger.debug("Background Index Update completed successfully.")
    except Exception as e:
        logger.exception(f"Failed to update index, {e}")
    finally:
        await shared_dict.put(key, False)


async def handle_update_index_request(
    http_request: Request, request: UpdateIndexRequest, background_tasks: BackgroundTasks
) -> UpdateIndexResponse:
    """
    Async handler for /update-index request.

    Args:
        request (QueryRequest): request payload

    Returns:
        UpdateIndexResponse: response content
    """
    logger, access_logger = get_heavyiq_logger(), _get_access_logger()
    access_logger.info("Request Received!", extra={"request": http_request})
    logger.debug("Document generation and index update request has been received!")
    shared_dict: SharedDictSingleton = SharedDictSingleton()
    key = "is_background_index_update_in_progress"

    if await shared_dict.get(key):
        logger.debug("Index Update is currently in-progress, skipping...")
        return UpdateIndexResponse(status=UpdateIndexStatus.in_progress.name)

    logger.debug("Started Index Update background task.")

    background_tasks.add_task(start_update_index_background_task, shared_dict, key, session=request.session_id)

    return UpdateIndexResponse(status=UpdateIndexStatus.started.name)
