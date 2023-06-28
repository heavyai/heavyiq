import json
from flask_sqlalchemy import SQLAlchemy
from langchain.schema import messages_from_dict, BaseMessage

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
    messages = db.Column(db.String, default="[]")

    def add_message(self, message: str):
        """
        Adds a single message to the messages list.
        """
        messages_list = json.loads(self.messages)
        messages_list.append(message)
        self.messages = json.dumps(messages_list)

    def get_chat_messages(self) -> list[BaseMessage]:
        """
        Converts self.messages into list of human or ai message.
        """
        messages_dict = []
        for i, message in enumerate(json.loads(self.messages)):
            if i % 2:
                # odd messages, ie. AI message
                messages_dict.append({"type": "ai", "data": {"content": message}})
            else:
                # even message, ie. human message
                messages_dict.append({"type": "human", "data": {"content": message}})

        return messages_from_dict(messages_dict)
