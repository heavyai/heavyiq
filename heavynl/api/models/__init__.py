import json
from flask_sqlalchemy import SQLAlchemy
from langchain.schema import messages_from_dict, BaseMessage, HumanMessage, AIMessage

db = SQLAlchemy()


class BaseModelMixin:
    """
    Base Mixin class for model classes.
    """

    def save(self):
        """
        Helps to save all the un-committed instance details into the database table.
        """
        db.session.add(self)
        db.session.commit()

    def delete(self):
        """
        Helps to delete the instance from database table.
        """
        db.session.delete(self)
        db.session.commit()


class SessionInfo(BaseModelMixin, db.Model):
    """
    SessionInfo table which stores chat history along with the session id.
    """

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(32))
    chat_messages = db.Column(db.String, default="[]")  # stores chat history
    sql_messages = db.Column(db.String, default="[]")  # stores sql history

    def add_chat_message(self, message: tuple[HumanMessage, AIMessage]):
        """
        Adds a tuple of langchain messages to the chat messages list.
        """
        question, answer = message
        messages_list = json.loads(self.chat_messages)
        messages_list.extend([question.content, answer.content])
        self.chat_messages = json.dumps(messages_list)

    def add_sql_message(self, message: tuple[HumanMessage, AIMessage]):
        """
        Adds a tuple of langchain messages to the sql messages list.
        """
        question, sql = message
        messages_list = json.loads(self.sql_messages)
        messages_list.extend([question.content, sql.content])
        self.sql_messages = json.dumps(messages_list)

    def _get_messages(self, messages_str: str) -> list[BaseMessage]:
        """
        Function which helps to recreate list of human or ai messages from db messages string.
        """
        messages_dict = []
        for i, message in enumerate(json.loads(messages_str)):
            if i % 2:
                # odd messages, ie. AI message
                messages_dict.append({"type": "ai", "data": {"content": message}})
            else:
                # even message, ie. human message
                messages_dict.append({"type": "human", "data": {"content": message}})

        return messages_from_dict(messages_dict)

    def get_chat_messages(self) -> list[BaseMessage]:
        """
        Converts chat_messages into list of human or ai message.
        """
        return self._get_messages(self.chat_messages)

    def get_sql_messages(self) -> list[BaseMessage]:
        """
        Converts sql_messages into list of human or ai message.
        """
        return self._get_messages(self.sql_messages)
