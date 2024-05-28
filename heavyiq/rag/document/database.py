import enum
import os
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, LargeBinary, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from heavyiq.config import get_config

CONFIG = get_config()


class Base(DeclarativeBase):
    pass


# Define the DocumentTypeEnum enumeration
class DocumentTypeEnum(enum.Enum):
    PDF = "pdf"
    JSON = "json"
    DOC = "doc"
    TXT = "text"

    @classmethod
    def get_by_content_type(cls: type["DocumentTypeEnum"], content_type: str) -> "DocumentTypeEnum":
        if "pdf" in content_type:
            return cls.PDF
        if "text" in content_type:
            return cls.TXT
        return cls.JSON


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, default=lambda: uuid.uuid4().hex, primary_key=True, unique=True, nullable=False)
    name = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=True)
    path = Column(String, nullable=True)
    collection_name = Column(String, nullable=False)
    document_type = Column(Enum(DocumentTypeEnum), nullable=False)  # type: ignore
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def serialize(self) -> dict:
        """
        Return serialized document.
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.document_type.value,
            "path": self.path,
            "created_at": str(self.created_at).split(".")[0],
        }


# Check if the directory exists, create it if necessary
os.makedirs(CONFIG.rag_documents_db_path, exist_ok=True)

DATABASE_URL = f"sqlite:///{CONFIG.rag_documents_db_path}/documents.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
