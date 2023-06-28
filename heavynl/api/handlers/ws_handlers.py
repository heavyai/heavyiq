from typing import Any
from flask import request, session
from heavynl.langchain.agents.convo_agent import create_conversational_agent
from heavynl.langchain.llms import get_chat_llm
from flask_socketio import Namespace, emit, join_room, leave_room, close_room, rooms, disconnect
from ..utils import generate_session_id


class ChatNamespace(Namespace):
    """
    Chat Namespace
    """

    def __init__(self, app, namespace=None):
        super().__init__(namespace)
        self.app = app

    def on_message(self, message: dict[str, Any]):
        emit("response", {"data": message["data"]})

    def on_question(self, message: dict[str, Any]):
        question = message["question"]
        sql_agent = create_conversational_agent()

        answer = sql_agent.run(input=question)
        emit("answer", {"data": answer})

    def on_connect(self, *args):
        session_id = session.get("session_id")

        if not session_id:
            # Generate a new session ID only if it doesn't exist
            session_id = generate_session_id()
            session["session_id"] = session_id
            session.modified = True

        print(session_id)
        join_room(session_id)

        print(f"Client connected: {session_id}")
        emit("setSession", {"data": session_id})

    def on_disconnect(self):
        session_id = session.get("session_id")  # Assuming the session ID is passed as a query parameter
        close_room(session_id)

        session.clear()

        print(f"Client disconnected: {session_id}")
