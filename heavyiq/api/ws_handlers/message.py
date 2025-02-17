from pydantic import BaseModel


# 🎯 Base WebSocket Message
class WSMessage(BaseModel):
    type: str
    message: str


#  SQL Message
class SQLMessage(WSMessage):
    type: str = "sql"


#  Processing Message
class ProcessingMessage(WSMessage):
    type: str = "processing"
    message: str = ""


#  Chart Message (Vega-Lite Spec)
class ChartMessage(WSMessage):
    type: str = "chart"
    message: dict
    new: bool = True


class TextMessage(WSMessage):
    type: str = "text"
    message: str
