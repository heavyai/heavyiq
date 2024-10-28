import uuid
from datetime import datetime
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy import Column, DateTime, String, Text, UniqueConstraint
from sqlalchemy import delete as sqldelete
from sqlalchemy import event, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from heavyrag.database.database import Base


class FactsModel(Base):
    """
    Table for storing facts about a database or facts about a table in database.
    """

    __tablename__ = "facts"

    id = Column(String, default=lambda: uuid.uuid4().hex, primary_key=True, unique=True, nullable=False)
    heavydb_name = Column(String, nullable=False)
    table_name = Column(String, nullable=True)
    fact = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False)

    __table_args__ = (UniqueConstraint("heavydb_name", "fact", name="uix_1"),)

    @classmethod
    def get(cls: type["FactsModel"], db_session: Session, id: str) -> Optional["FactsModel"]:
        """
        Get by facts object.
        """
        return db_session.query(cls).get(id)

    def serialize(self) -> dict:
        """
        Serializes the current object.
        """
        return {"id": self.id, "fact": self.fact, "created_at": self.created_at, "updated_at": self.updated_at}

    @classmethod
    def list(
        cls: type["FactsModel"], db_session: Session, heavydb_name: str, serialize: bool = False
    ) -> list["FactsModel"] | list[dict]:
        """
        Get by heavydb name.
        """
        result = db_session.query(cls).filter_by(heavydb_name=heavydb_name).all()
        if serialize:
            return [i.serialize() for i in result]
        return result

    @classmethod
    def update(cls: type["FactsModel"], db_session: Session, id: str, fact: str) -> str | None:
        """
        Update database fact.
        """
        stmt = update(cls).where(cls.id == id).values(fact=fact)
        # Execute the update statement
        db_session.execute(stmt)
        db_session.commit()
        # return the id which got updated, if there isn't any update happens
        # then a None value should be returned
        return id  # type: ignore

    @classmethod
    def add(
        cls: type["FactsModel"],
        db_session: Session,
        heavydb_name: str,
        fact: str,
    ) -> str:
        """
        Helps to add or update facts relevant to a database.
        """
        stmt = insert(cls).values(heavydb_name=heavydb_name, fact=fact)
        result = db_session.execute(stmt)
        inserted_id = result.inserted_primary_key[0]
        db_session.commit()

        return inserted_id

    @classmethod
    def bulk_insert(cls: type["FactsModel"], db_session: Session, heavydb_name: str, facts: List[str]) -> List[str]:
        """
        Bulk insert facts.
        """
        # This works on all envs except for RHEL where we need to install pysqlite3 binary manually
        # which does not support returning of ids upon bulk insertion
        # so do a simple insertion of each snippet iteratively and return the final ids
        # ids = db_session.scalars(
        #     insert(cls).values(heavydb_name=heavydb_name).returning(cls.id, sort_by_parameter_order=True),
        #     [{"fact": fact} for fact in facts],
        # ).all()
        # db_session.commit()
        ids = []
        for fact in facts:
            id_ = cls.add(db_session=db_session, heavydb_name=heavydb_name, fact=fact)
            ids.append(id_)
        return ids  # type: ignore

    @classmethod
    def delete(cls: type["FactsModel"], db_session: Session, ids: List[str]) -> bool:
        """
        Delete facts.
        """
        assert ids
        # Query to count existing records with the specified IDs before deletion
        existing_count = db_session.query(cls).filter(cls.id.in_(ids)).count()

        # Execute the delete statement without RETURNING
        stmt = sqldelete(cls).where(cls.id.in_(ids))
        db_session.execute(stmt, execution_options={"synchronize_session": "fetch"})

        # Check if all records were deleted by counting remaining rows with those IDs
        remaining_count = db_session.query(cls).filter(cls.id.in_(ids)).count()

        has_deleted = (existing_count - remaining_count) == len(ids)
        if has_deleted:
            # Commit only if all specified IDs were successfully deleted
            db_session.commit()

        return has_deleted

    @classmethod
    def delete_facts_by_database(cls: type["FactsModel"], db_session: Session, heavydb_name: str) -> None:
        """
        Delete all facts relevant to a database.
        """
        stmt = sqldelete(cls).where(cls.heavydb_name == heavydb_name)
        db_session.execute(stmt)
        db_session.commit()


# Custom exception class
class RAGDBIntegrityError(Exception):
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception
