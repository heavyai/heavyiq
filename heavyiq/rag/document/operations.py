import tempfile
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm.session import Session

from .database import Document, DocumentTypeEnum


async def insert_document(
    session: Session, heavydb_name: str, file: UploadFile | None = None, file_path: str | None = None
):
    """
    Inserts document into sqlite documents table.
    """
    file_content, filename = None, None
    if file:
        # Read the content of the file
        file_content = await file.read()
        filename = file.filename
        document_type = DocumentTypeEnum.get_by_content_type(file.content_type)  # type: ignore
    elif file_path:
        filename = Path(file_path).name
        document_type = DocumentTypeEnum.get_by_content_type(filename)
    else:
        raise ValueError("Atleast file or file_path should be passed!")

    # check for document exists
    document = session.query(Document).filter_by(name=filename, collection_name=heavydb_name).first()
    if document:
        return None

    # Create a Document object
    document = Document(
        name=filename, content=file_content, path=file_path, document_type=document_type, collection_name=heavydb_name
    )

    # Add the Document object to the session and commit changes
    session.add(document)
    session.commit()


def retrieve_document(document_id: str, session: Session):
    """
    Retrieve document from sqlite documents table.
    """
    # Retrieve the Document object by its ID
    document = session.query(Document).filter_by(id=document_id).first()

    if document:
        # Write the file content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(document.content)
        temp_file.close()
        return temp_file.name
    else:
        return None


def list_documents(session: Session, dbname: str) -> list[Document]:
    """
    List all the uploaded documents.
    """
    return [i for i in session.query(Document).filter_by(collection_name=dbname).all()]


def list_documents_serialized(session: Session, dbname: str) -> list[dict]:
    """
    List all the uploaded documents.
    """
    return [i.serialize() for i in session.query(Document).filter_by(collection_name=dbname).all()]


def delete_document(document_id: str, collection_name: str, session: Session) -> None:
    """
    Deletes a document by it's id.
    """
    document = session.query(Document).filter_by(id=document_id, collection_name=collection_name).first()

    if document:
        # Write the file content to a temporary file
        session.delete(document)
        session.commit()
    else:
        return None
