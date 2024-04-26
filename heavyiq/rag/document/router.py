import re

from fastapi import APIRouter, Body, Depends, Form, UploadFile
from heavyrag.documents.etl import DocumentIndexETL
from heavyrag.documents.index import delete_nodes_by_database_doc_id
from heavyrag.documents.ingest import sync_doc_index
from heavyrag.documents.read import DocumentDatabaseReader
from sqlalchemy.orm import Session as Session

from heavyiq.langchain import HeavyDB
from heavyiq.rag.document.database import Session as DBSession
from heavyiq.rag.document.database import engine
from heavyiq.rag.document.models import (
    AskDocumentRequest,
    AskDocumentResponse,
    BaseModelWithSessionID,
    ListFilesResponse,
)
from heavyiq.rag.document.operations import delete_document, insert_document, list_documents_serialized

doc_router = APIRouter()


# Dependency function to provide a SQLAlchemy session
def get_db():
    try:
        db = DBSession()
        yield db
    finally:
        db.close()


# dependency function which provides HeavyDB instance
async def get_current_heavydb_instance(request: BaseModelWithSessionID = Body(...)) -> HeavyDB:
    """Dependency which helps to get the HeavyDB instance"""
    return await HeavyDB.from_session_async(request.session_id)


async def get_collection_name(heavydb: HeavyDB = Depends(get_current_heavydb_instance)) -> str:
    """Dependency which helps to get the collection name (ie. heavydb database name)"""
    return heavydb._dbname


@doc_router.post("/uploadfiles/")
async def upload_files(
    files: list[UploadFile],
    # request: UploadFilesParameters,
    session_id: str = Form(...),
    session: Session = Depends(get_db),
) -> None:
    heavydb = await HeavyDB.from_session_async(session_id)
    collection_name = heavydb._dbname
    for file in files:
        await insert_document(session, collection_name, file=file)


@doc_router.post("/files/", response_model=ListFilesResponse)
async def list_files(
    session: Session = Depends(get_db),
    collection_name: HeavyDB = Depends(get_collection_name),
) -> ListFilesResponse:
    return ListFilesResponse(documents=list_documents_serialized(session, collection_name))


@doc_router.delete("/files/delete/{doc_id}")
async def delete_file(
    doc_id: str, session: Session = Depends(get_db), collection_name: str = Depends(get_collection_name)
) -> dict[str, str]:
    """
    Helps to delete a document from documents DB and also it's embeddings from vectordb.
    """
    try:
        delete_nodes_by_database_doc_id(doc_id, collection_name=collection_name)
    except Exception as e:
        raise e
    else:
        delete_document(document_id=doc_id, collection_name=collection_name, session=session)

    return {"message": f"Item {doc_id} has been deleted"}


@doc_router.post("/sync_index/")
async def sync_index(
    session: Session = Depends(get_db), collection_name: str = Depends(get_collection_name)
) -> dict[str, str]:
    """
    Helps to sync the doucment index.
    This involves cheking of any newly added documents on documents table and then
    it tries to transform each document to nodes, finally add those
    nodes into the index.
    """
    reader = DocumentDatabaseReader(engine=engine)
    sync_doc_index(reader, collection_name)
    return {"status": "succcess"}


@doc_router.post("/ask", response_model=AskDocumentResponse)
async def ask(request: AskDocumentRequest, collection_name: str = Depends(get_collection_name)) -> AskDocumentResponse:
    """
    Ask questions regrading the uploaded docs.
    """
    etl = DocumentIndexETL(
        reader_cls=DocumentDatabaseReader,
        reader_init_kwargs={"engine": engine},
    )
    etl.collection_name = collection_name
    response = etl.run(request.question)
    output, metadata, source_nodes = response.response, response.metadata or {}, response.source_nodes
    output = re.sub(r"^Response\s*\d+\s*:\s*", "", output)

    return AskDocumentResponse(answer=output, metadata=metadata, sources=[i.get_content() for i in source_nodes])
