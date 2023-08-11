import asyncio
import json
from collections.abc import Awaitable
from typing import Any, Iterable, Iterator
from fastapi_socketio import SocketManager
from langchain.schema import messages_from_dict, HumanMessage, AIMessage, BaseMessage


class AsyncIteratorWrapper:
    """The following is a utility class that transforms a
    regular iterable to an asynchronous one.

    link: https://www.python.org/dev/peps/pep-0492/#example-2
    """

    _it: Iterator[Any]

    def __init__(self, obj: Iterable[Any]) -> None:
        self._it = iter(obj)

    def __aiter__(self) -> "AsyncIteratorWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


async def wrap_done(fn: Awaitable, event: asyncio.Event) -> Any:
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        return await fn
    except Exception as e:
        print(f"Caught exception: {e}")
        raise e
    finally:
        # Signal the event/aiter to stop.
        event.set()


class MessageHistoryManager:
    """
    class responsible for saving and retrieving message history to/from python socket-io session.
    """

    chat_history_key: str = "chat_history"
    sql_history_key: str = "sql_history"

    def __init__(self, socket_manager: SocketManager, sid: str) -> None:
        self.socket_manager = socket_manager
        self.sid = sid

    async def save_message_to_session(self, question: HumanMessage, answer: AIMessage, key: str):
        """
        Converts langchain messages to json and store it as json object on session.

        Args:
            socket_manager (SocketManager): fastapi socketmanager instance
            sid (str): session id
            question (HumanMessage): Question asked
            answer (AIMessage): Answer replied
            key (str): key to access/store the messages from/to session
        """
        # check for any values exists on the appropriate key
        # if yes, then append the current message to the existsing list else add it as new

        session_info = await self.socket_manager.get_session(self.sid)
        key_info = session_info.get(key)
        messages: list[str] = []
        if key_info:
            # key already exists, so load it as json
            messages: list[str] = json.loads(key_info)
        messages.extend([question.content, answer.content])
        # alter session info with the new values for the passed key
        session_info[key] = json.dumps(messages)
        # save the session with modified info
        await self.socket_manager.save_session(self.sid, session_info)

    async def save_chat_history(self, question: HumanMessage, answer: AIMessage):
        """
        Saves all the chat history messages to session.
        """
        await self.save_message_to_session(question=question, answer=answer, key=self.chat_history_key)

    async def save_sql_history(self, question: HumanMessage, answer: AIMessage):
        """
        Saves all the sql history (n successful sql queries) messages to session.
        """
        await self.save_message_to_session(question=question, answer=answer, key=self.sql_history_key)

    async def get_messages_from_session(self, key: str) -> list[BaseMessage]:
        """
        Get all the messages specific to a particular key from session.
        """
        session_info = await self.socket_manager.get_session(self.sid)
        history = session_info.get(key)
        if not history:
            return []
        messages_dict = []
        for i, message in enumerate(json.loads(history)):
            if i % 2:
                # odd messages, ie. AI message
                messages_dict.append({"type": "ai", "data": {"content": message}})
            else:
                # even message, ie. human message
                messages_dict.append({"type": "human", "data": {"content": message}})

        return messages_from_dict(messages_dict)

    async def get_chat_history(self) -> list[BaseMessage]:
        return await self.get_messages_from_session(self.chat_history_key)

    async def get_sql_history(self) -> list[BaseMessage]:
        return await self.get_messages_from_session(self.sql_history_key)
