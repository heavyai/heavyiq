from pathlib import Path

import sqlalchemy.exc as sqlalchemy_exc
from fastapi import APIRouter, Body, Depends, Form, HTTPException, UploadFile

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.rag import models as md
from heavyrag.database import FactsModel, RAGDBIntegrityError, ragdb

CONFIG = get_config()
doc_router = APIRouter()
logger = get_heavyiq_logger()


# dependency function which provides HeavyDB instance
async def get_current_heavydb_instance(request: md.BaseModelWithSessionID = Body(...)) -> HeavyDB:
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


@doc_router.post("/list", response_model=md.ListFilesResponse)
async def list_files(
    collection_name: HeavyDB = Depends(get_collection_name),
) -> md.ListFilesResponse:
    source_dir = f"{CONFIG.rag_documents_source_dir}/{collection_name}"
    return md.ListFilesResponse(documents=[fil.name for fil in Path(source_dir).glob("*") if fil.is_file()])


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


@doc_router.post("/ask", response_model=md.AskDocumentResponse)
async def ask_about_document(
    request: md.AskDocumentRequest, collection_name: str = Depends(get_collection_name)
) -> md.AskDocumentResponse:
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

    return md.AskDocumentResponse(
        answer=response.response,
        metadata=response.metadata,
        sources=[i.get_content(metadata_mode="all") for i in response.source_nodes],
        passing=passing,
        score=score,
    )


table_router = APIRouter()


@table_router.post("/names")
async def derive_table_names(
    request: md.AskDocumentRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> list[str]:
    """
    Retrieve table names which are relevant to the asked question.
    """
    from heavyrag.main import determine_table_names

    return await determine_table_names(question=request.question, heavydb=heavydb)


@table_router.post("/index/nodes")
async def get_table_nodes(
    request: md.GetSnippetsfromIndexRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.GetSnippetsfromIndexResponse:
    """
    Get facts from vectorDB index.
    """
    from heavyrag.main import get_table_nodes

    nodes = await get_table_nodes(heavydb_name=heavydb._dbname, limit=request.limit)
    return md.GetSnippetsfromIndexResponse(
        nodes=[md.GetSnippetsfromIndexResponse.Node(content=i.get_text(), metadata=i.metadata) for i in nodes]
    )


# RAG database router
# which helps to make CRUD operations on RAG database, especially for facts
facts_db_router = APIRouter()


# insert fact
@facts_db_router.post("/insert")
async def add_snippet(
    request: md.AddSnippetRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.AddSnippetResponse:
    """
    Endpoint for adding a new snippet.
    """
    from heavyrag.ingest import ainsert_fact

    with ragdb.get_db() as session:
        fact_id = None
        try:
            # add db record
            fact_id = FactsModel.add(db_session=session, heavydb_name=heavydb._dbname, fact=request.snippet)
            # add to index
            await ainsert_fact(fact_id=fact_id, fact=request.snippet, heavydb_name=heavydb._dbname)

            return md.AddSnippetResponse(snippet_id=fact_id)
        except sqlalchemy_exc.IntegrityError as e:
            raise RAGDBIntegrityError("UNIQUE constraint failed on fact", e)
        except Exception as e:
            if fact_id:
                FactsModel.delete(db_session=session, ids=[fact_id])
            raise e


# bulk insert facts
@facts_db_router.post("/bulk-insert")
async def bulk_insert_snippets(
    request: md.BulkInsertSnippetsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.BulkInsertSnippetsResponse:
    """
    Bulk insert snippets into sqlite database and store the relavant embeddings on vector database.
    """
    from heavyrag.ingest import ainsert_facts

    with ragdb.get_db() as session:
        fact_ids: list[str] = []
        try:
            # add db record
            fact_ids = FactsModel.bulk_insert(db_session=session, heavydb_name=heavydb._dbname, facts=request.snippets)
            # add to index
            await ainsert_facts(facts=list(zip(fact_ids, request.snippets)), heavydb_name=heavydb._dbname)

            return md.BulkInsertSnippetsResponse(snippet_ids=fact_ids)
        except sqlalchemy_exc.IntegrityError as e:
            raise RAGDBIntegrityError("UNIQUE constraint failed on fact", e)
        except Exception as e:
            if fact_ids:
                FactsModel.delete(db_session=session, ids=fact_ids)
            raise e


# update fact
@facts_db_router.post("/update")
async def update_snippet(
    request: md.UpdateSnippetRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.AddSnippetResponse:
    """
    Updates a particular snippet by snippet_id.
    """
    from heavyrag.ingest import aupdate_fact

    with ragdb.get_db() as session:
        try:
            # add db record
            updated_id = FactsModel.update(db_session=session, id=request.snippet_id, fact=request.snippet)
            if not updated_id:
                raise ValueError(f"Failed to update fact id, {request.snippet_id}")
            # add to index
            await aupdate_fact(fact_id=request.snippet_id, fact=request.snippet, heavydb_name=heavydb._dbname)

            return md.AddSnippetResponse(snippet_id=request.snippet_id)
        except sqlalchemy_exc.IntegrityError as e:
            raise RAGDBIntegrityError("UNIQUE constraint failed on fact", e)

        except Exception as e:
            raise e


@facts_db_router.post("/get")
async def get_snippet(
    request: md.GetSnippetRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.SnippetResponse:
    """
    Return a particular snippet by snippet_id.
    """
    with ragdb.get_db() as session:
        fact = FactsModel.get(db_session=session, id=request.snippet_id)
        if fact and fact.heavydb_name == heavydb._dbname:
            return md.SnippetResponse(
                snippet=fact.fact, snippet_id=fact.id, created_at=fact.created_at, updated_at=fact.updated_at
            )
        raise HTTPException(status_code=404)


@facts_db_router.post("/delete")
async def delete_snippets(
    request: md.DeleteSnippetsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.DeleteSnippetResponse:
    """
    Delete a particular database fact.
    """
    from heavyrag.ingest import adelete_facts

    with ragdb.get_db() as session:
        try:
            deleted = FactsModel.delete(db_session=session, ids=request.snippet_ids)
            if deleted:
                await adelete_facts(heavydb_name=heavydb._dbname, fact_ids=request.snippet_ids)
                return md.DeleteSnippetResponse(deleted=True)
            return md.DeleteSnippetResponse(deleted=False)

        except Exception as e:
            raise e


@facts_db_router.post("/list")
async def list_snippets(
    request: md.ListSnippetsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.ListSnippetsResponse:
    """
    List all snippets associated with a database.
    """
    with ragdb.get_db() as session:
        facts = FactsModel.list(db_session=session, heavydb_name=heavydb._dbname, serialize=True)
        snippets = [
            {"snippet_id": i["id"], "snippet": i["fact"], "created_at": i["created_at"], "updated_at": i["updated_at"]}
            for i in facts
        ]
        return md.ListSnippetsResponse(snippets=snippets)


@facts_db_router.post("/delete_all")
async def delete_all_snippets(
    request: md.DeleteAllSnippetsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.DeleteSnippetResponse:
    """
    Delete all snippets asscoiated with a database. This involves deleteing database entries and index nodes.
    """
    from heavyrag.ingest import adelete_facts

    with ragdb.get_db() as session:
        try:
            FactsModel.delete_facts_by_database(db_session=session, heavydb_name=heavydb._dbname)
            await adelete_facts(heavydb_name=heavydb._dbname)
            return md.DeleteSnippetResponse(deleted=True)
        except Exception as e:
            raise e


@facts_db_router.post("/index/nodes")
async def get_snippet_nodes(
    request: md.GetSnippetsfromIndexRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.GetSnippetsfromIndexResponse:
    """
    Get facts from vectorDB index.
    """
    from heavyrag.main import get_facts

    nodes = await get_facts(heavydb_name=heavydb._dbname, limit=request.limit)
    return md.GetSnippetsfromIndexResponse(
        nodes=[md.GetSnippetsfromIndexResponse.Node(content=i.get_text(), metadata=i.metadata) for i in nodes]
    )


@facts_db_router.post("/ask", response_model=md.AskSnippetsResponse)
async def ask_about_snippets(
    request: md.AskSnippetsRequest, collection_name: str = Depends(get_collection_name)
) -> md.AskSnippetsResponse:
    """
    Ask questions regrading the uploaded snippets.
    """
    from heavyrag.main import ask_facts

    out = await ask_facts(
        question=request.question,
        heavydb_name=collection_name,
        only_retrieve=request.only_retrieve,
        do_evaluate=request.do_evaluate,
        similarity_cutoff=CONFIG.rag_facts_similarity_cutoff_score,
        similarity_top_k=CONFIG.rag_facts_similarity_top_k,
        with_reranker=request.use_reranker,
        reranker_top_k=CONFIG.rag_facts_reranker_top_k,
        reranker_cutoff=CONFIG.rag_facts_reranker_cutoff_score,
    )
    if request.only_retrieve:
        return md.AskSnippetsResponse(sources=out)

    response, eval_result = out
    passing, score = None, None
    if eval_result:
        passing = eval_result.passing
        score = eval_result.score

    return md.AskSnippetsResponse(
        answer=response.response,
        metadata=response.metadata,
        sources=[i.get_content(metadata_mode="all") for i in response.source_nodes],
        passing=passing,
        score=score,
    )


collection_router = APIRouter()


@collection_router.post("/re-embed", response_model=md.ReEmbedDocumentsResponse)
def reembed_documents_in_chroma_collection(request: md.ReEmbedDocumentsRequest) -> md.ReEmbedDocumentsResponse:
    """
    re-embed docs.
    """
    from heavyrag.ingest import re_embed_documents

    success = re_embed_documents(request.collection_name)
    return md.ReEmbedDocumentsResponse(success=success)
