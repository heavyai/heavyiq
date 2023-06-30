from typing import Any

from flask import request
from flask_socketio import Namespace, close_room, emit, join_room
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

from heavynl.logging_utils import get_heavynl_logger
from heavynl.api.models import SessionInfo
from heavynl.config import get_config
from heavynl.langchain.llms import get_chat_llm
from heavynl.langchain.agents.convo_agent import create_conversational_agent
from heavynl.langchain.callbacks import StreamingChatCallbackHandler


class ChatNamespace(Namespace):
    """
    All the chat related web socket requests are handled here.
    """

    def on_question(self, message: dict[str, Any]):
        """
        Handles all the question events emitted by WebSocket client.
        """
        question = message["question"]
        session_id = message["sessionID"]
        user_session = SessionInfo.query.filter_by(session_id=session_id).first()
        retrieved_chat_history = ChatMessageHistory(messages=user_session.get_chat_messages())
        retrieved_sql_history = ChatMessageHistory(messages=user_session.get_sql_messages())

        stream_handler = StreamingChatCallbackHandler()
        chat_llm = get_chat_llm(
            ["agent", "conversational_sql_agent"],
            temperature=0,
            model=get_config().openai_gpt_model,
            streaming=True,
            callbacks=[stream_handler],
        )
        sql_agent = create_conversational_agent(
            chat_llm=chat_llm, chat_history=retrieved_chat_history, sql_history=retrieved_sql_history, verbose=True
        )

        # generate answer
        answer = sql_agent.run(question)
        try:
            # save the chat and sql history along with the question and session id to SessionInfo db table.
            message_history = sql_agent.memory.load_memory_variables({})
            chat_message_history, sql_message_history = message_history["chat_history"], message_history["last_run_sql"]
            user_session.add_chat_message((chat_message_history[-2], chat_message_history[-1]))
            if sql_message_history:
                user_session.add_sql_message((sql_message_history[-2], sql_message_history[-1]))

            user_session.save()
        except Exception as e:
            get_heavynl_logger().error(f"Failed to save chat and sql history, {e}")
        # emits answer event to WS client for displaying
        emit("answer", {"data": answer})
        emit("endToken", {"data": answer})

    def on_connect(self):
        """
        Triggerd immediately upon ws client connect event.
        """
        session_id = request.sid  # type: ignore
        # store the session id to db
        user_session = SessionInfo(session_id=session_id)  # type: ignore
        user_session.save()

        # join chat room with the session id as room name
        join_room(session_id)

        print(f"Client connected: {session_id}")
        # emits setSession event, so that the client can store the session id to localstorage
        emit("setSession", {"data": session_id})

    def on_disconnect(self):
        """
        Triggerd immediately upon ws client disconnect event like browser tab, window close events.
        """
        session_id = request.sid  # type: ignore
        close_room(session_id)

        # delete all the data relevevant to the particular session id upon diconnect
        user_session = SessionInfo.query.filter_by(session_id=session_id).first()
        if user_session:
            user_session.delete()

        print(f"Client disconnected: {session_id}")
