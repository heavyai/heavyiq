import uuid
from typing import Optional

from sqlalchemy import Column, String, Text, exc
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from heavyrag.database.database import Base


class FactsModel(Base):
    """
    Table for storing facts about a database or facts about a table in database.
    """

    __tablename__ = "facts"

    id = Column(String, default=lambda: uuid.uuid4().hex, primary_key=True, unique=True, nullable=False)
    heavydb_name = Column(String, index=True, unique=True, nullable=False)
    table_name = Column(String, nullable=True)
    facts = Column(Text, nullable=True)

    @classmethod
    def get(cls: type["FactsModel"], db_session: Session, heavydb_name: str) -> Optional["FactsModel"]:
        """
        Get by heavydb name.
        """
        try:
            facts = db_session.query(cls).filter_by(heavydb_name=heavydb_name).one()
        except exc.NoResultFound:
            return None
        else:
            return facts

    @classmethod
    def add_or_update_database_facts(
        cls: type["FactsModel"],
        db_session: Session,
        heavydb_name: str,
        facts: str,
    ) -> "FactsModel":
        """
        Helps to add or update facts relevant to a database.
        """
        stmt = insert(cls).values(heavydb_name=heavydb_name, facts=facts)
        stmt = stmt.on_conflict_do_update(
            index_elements=["heavydb_name"],
            where=(cls.heavydb_name == heavydb_name),
            set_=dict(facts=facts, heavydb_name=stmt.excluded.heavydb_name),
        )
        db_session.execute(stmt)
        db_session.commit()

        return cls.get(db_session=db_session, heavydb_name=heavydb_name)

    @classmethod
    def delete(cls: type["FactsModel"], db_session: Session, heavydb_name: str) -> bool:
        """
        Delete facts.
        """
        record = cls.get(db_session=db_session, heavydb_name=heavydb_name)
        if record:
            db_session.delete(record)
            db_session.commit()
            return True

        return False
