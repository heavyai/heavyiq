import uuid

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Enum, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from .enums import LangChainType

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "call_log"

    id = Column(String, default=lambda: uuid.uuid4().hex, primary_key=True, unique=True, nullable=False)
    langchain_type = Column(Enum(LangChainType), nullable=False)
    langchain_name = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    input = Column(JSON, nullable=False)
    output = Column(JSON, nullable=True)  # null if error
    model = Column(String, nullable=False)
    error = Column(String, nullable=True)
    agent_log = Column(String, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)  # potentially null if error
    completion_tokens = Column(Integer, nullable=True)  # potentially null if error
    total_tokens = Column(Integer, nullable=True)  # potentially null if error
    successful_requests = Column(Integer, nullable=True)  # potentially null if error
    total_cost = Column(Float, nullable=True)  # potentially null if error


engine = create_engine("sqlite:///langchain_call_log.db")
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
