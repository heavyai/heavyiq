import asyncio
from typing import Any

from fastapi_socketio import SocketManager
from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from heavyiq.api.utils import MessageHistoryManager, wrap_done_iter
from heavyiq.config import get_config
from heavyiq.langchain.agents.convo_agent import create_conversational_agent_async
from heavyiq.langchain.llms import get_chat_llm
from heavyiq.logging_utils import get_heavyiq_logger


def register_socket_events(socket_manager: SocketManager):
    @socket_manager.on("connect")  # type: ignore
    async def handle_connect(sid: str, environ: dict[str, Any]):
        """
        ws connect handler.
        """
        ...

    @socket_manager.on("question")  # type: ignore
    async def handle_question(sid: str, message: dict[str, str]):
        """
        socket-io question handler function.

        Args:
            sid (str): python socketio session id.
            message (dict[str, str]): input dict which contains question.
        """
        question = message["question"]
        # define file logger
        logger = get_heavyiq_logger()
        file_callback_handler = logger.async_langchain_cb_handler(to_stdout=False)
        message_history_manager = MessageHistoryManager(socket_manager=socket_manager, sid=sid)
        # retrieve chat/sql history from current session
        retrieved_chat_history = ChatMessageHistory(messages=await message_history_manager.get_chat_history())
        retrieved_sql_history = ChatMessageHistory(messages=await message_history_manager.get_sql_history())

        stream_handler = AsyncFinalIteratorCallbackHandler()
        chat_llm = get_chat_llm(
            tags=["agent", "conversational_sql_agent"],
            temperature=0,
            streaming=True,
            model=get_config().openai_gpt_model,
            callbacks=[stream_handler],
        )
        sql_agent = await create_conversational_agent_async(
            chat_llm=chat_llm,
            chat_history=retrieved_chat_history,
            sql_history=retrieved_sql_history,
            verbose=True,
        )
        # Begin a task that runs in the background.
        task = asyncio.create_task(
            wrap_done_iter(
                sql_agent.iter(question, callbacks=[file_callback_handler], async_=True),
                stream_handler.done,
                socket_manager=socket_manager,
                sid=sid,
            ),
        )
        # tell the FE that the token sending is in progress
        await socket_manager.emit("startToken", {})

        # watch for token arrival
        async for token in stream_handler.aiter():
            # emit newToken on every new token arrives
            await socket_manager.emit("newToken", {"data": token})

        answer = await task
        # load memory variables and save the history data to current session
        message_history = sql_agent.memory.load_memory_variables({})  # type: ignore
        chat_message_history, sql_message_history = message_history["chat_history"], message_history["last_run_sql"]

        await message_history_manager.save_chat_history(
            question=chat_message_history[-2], answer=chat_message_history[-1]
        )
        if sql_message_history:
            await message_history_manager.save_sql_history(
                question=sql_message_history[-2], answer=sql_message_history[-1]
            )
        # task finished , send the endtoken as well as the final answer
        await socket_manager.emit("answer", {"data": answer})
        await socket_manager.emit("endToken", {})
