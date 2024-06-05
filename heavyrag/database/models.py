import uuid
from typing import Optional

from sqlalchemy import Column, String, Text
from sqlalchemy import delete as sqldelete
from sqlalchemy import update
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
        return {"id": self.id, "fact": self.fact}

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
        stmt = update(cls).returning(cls.id).where(cls.id == id).values(fact=fact)
        # Execute the update statement
        result = db_session.execute(stmt)
        updated_id = result.fetchone()
        db_session.commit()
        # return the id which got updated, if there isn't any update happens
        # then a None value should be returned
        return updated_id  # type: ignore

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
    def delete(cls: type["FactsModel"], db_session: Session, id: str) -> bool:
        """
        Delete facts.
        """
        record = cls.get(db_session=db_session, id=id)
        if record:
            db_session.delete(record)
            db_session.commit()
            return True

        return False

    @classmethod
    def delete_facts_by_database(cls: type["FactsModel"], db_session: Session, heavydb_name: str) -> None:
        """
        Delete all facts relevant to a database.
        """
        stmt = sqldelete(cls).where(cls.heavydb_name == heavydb_name)
        db_session.execute(stmt)
        db_session.commit()
