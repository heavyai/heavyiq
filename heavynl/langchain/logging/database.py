import uuid

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Enum, Float, JSON, Table, MetaData
from .enums import LangChainType

request_log_metadata = MetaData()

request_log_table = Table(
    "call_log",
    request_log_metadata,
    Column("id", String, default=lambda: uuid.uuid4().hex, primary_key=True, unique=True, nullable=False),
    Column("langchain_type", Enum(LangChainType), nullable=False),
    Column("langchain_name", String, nullable=False),
    Column("start_time", DateTime, nullable=False),
    Column("end_time", DateTime, nullable=False),
    Column("input", JSON, nullable=False),
    Column("output", JSON, nullable=True),  # null if error
    Column("model", String, nullable=False),
    Column("error", String, nullable=True),
    Column("agent_log", String, nullable=True),
    Column("prompt_tokens", Integer, nullable=True),  # potentially null if error
    Column("completion_tokens", Integer, nullable=True),  # potentially null if error
    Column("total_tokens", Integer, nullable=True),  # potentially null if error
    Column("successful_requests", Integer, nullable=True),  # potentially null if error
    Column("total_cost", Float, nullable=True),  # potentially null if error
)

engine = create_engine("sqlite:///langchain_call_log.db")
request_log_metadata.create_all(engine)
