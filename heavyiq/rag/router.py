import re
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session as Session

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.rag.models import (
    AskDocumentRequest,
    AskDocumentResponse,
    AskFactsRequest,
    AskFactsResponse,
    BaseModelWithSessionID,
    ListFilesResponse,
)

CONFIG = get_config()
doc_router = APIRouter()


# dependency function which provides HeavyDB instance
async def get_current_heavydb_instance(request: BaseModelWithSessionID = Body(...)) -> HeavyDB:
    """Dependency which helps to get the HeavyDB instance"""
    return await HeavyDB.from_session_async(request.session_id)


async def get_collection_name(heavydb: HeavyDB = Depends(get_current_heavydb_instance)) -> str:
    """Dependency which helps to get the collection name (ie. heavydb database name)"""
    return heavydb._dbname


@doc_router.post("/upload")
async def upload_files(
    files: list[UploadFile],
    session_id: str = Form(...),
) -> list[str]:
    """
    Supposed to save files into the configured source directory.
    """
    from heavyrag.ingest import ainsert_document

    heavydb = await HeavyDB.from_session_async(session_id)
    collection_name = heavydb._dbname
    source_dir = f"{CONFIG.rag_documents_source_dir}/{collection_name}"
    # Create directory if it doesn't exist
    Path(source_dir).mkdir(parents=True, exist_ok=True)
    uploaded_files = []
    for file in files:
        target_file_path = f"{source_dir}/{file.filename}"
        try:
            # Save the file locally
            with open(target_file_path, "wb") as file_object:
                file_object.write(file.file.read())
                uploaded_files.append(file.filename)

            # insert the document into VectorDB
            await ainsert_document(target_file_path, heavydb_name=collection_name)
        except Exception as e:
            # any exception then delete the saved file
            print(f"Failed to upload document, {e}")
            Path(target_file_path).unlink(missing_ok=True)

    return uploaded_files


@doc_router.post("/list", response_model=ListFilesResponse)
async def list_files(
    collection_name: HeavyDB = Depends(get_collection_name),
) -> ListFilesResponse:
    source_dir = f"{CONFIG.rag_documents_source_dir}/{collection_name}"
    return ListFilesResponse(documents=[fil.name for fil in Path(source_dir).glob("*") if fil.is_file()])


@doc_router.delete("/delete")
async def delete_file(file_name: str, collection_name: str = Depends(get_collection_name)) -> dict[str, str]:
    """
    Helps to delete a document from documents DB and also it's embeddings from vectordb.
    """
    from heavyrag.ingest import adelete_document

    try:
        # delete relevant nodes
        await adelete_document(file_name=file_name, heavydb_name=collection_name)
        source_dir = f"{CONFIG.rag_documents_source_dir}/{collection_name}/{file_name}"
        # delete actual file
        Path(source_dir).unlink()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{file_name} not found!")

    return {"message": f"Document {file_name} has been deleted."}


@doc_router.post("/ask", response_model=AskDocumentResponse)
async def ask_about_document(
    request: AskDocumentRequest, collection_name: str = Depends(get_collection_name)
) -> AskDocumentResponse:
    """
    Ask questions regrading the uploaded docs.
    """
    from heavyrag.main import ask_document

    response, eval_result = await ask_document(
        question=request.question, heavydb_name=collection_name, do_evaluate=True
    )
    passing, score = None, None
    if eval_result:
        passing = eval_result.passing
        score = eval_result.score

    return AskDocumentResponse(
        answer=response.response,
        metadata=response.metadata,
        sources=[i.get_content(metadata_mode="all") for i in response.source_nodes],
        passing=passing,
        score=score,
    )


@doc_router.post("/ask_facts", response_model=AskFactsResponse)
async def ask_about_facts(
    request: AskFactsRequest, collection_name: str = Depends(get_collection_name)
) -> AskFactsResponse:
    """
    Ask questions regrading the uploaded docs.
    """
    from heavyrag.main import ask_facts

    response, eval_result = await ask_facts(question=request.question, heavydb_name=collection_name, do_evaluate=True)
    passing, score = None, None
    if eval_result:
        passing = eval_result.passing
        score = eval_result.score

    return AskFactsResponse(
        answer=response.response,
        metadata=response.metadata,
        sources=[i.get_content(metadata_mode="all") for i in response.source_nodes],
        passing=passing,
        score=score,
    )


table_router = APIRouter()


@table_router.post("/names")
async def derive_table_names(
    request: AskDocumentRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> list[str]:
    """
    Retrieve table names which are relevant to the asked question.
    """
    from heavyrag.main import determine_table_names

    return await determine_table_names(question=request.question, heavydb=heavydb)
