from pathlib import Path

from fastapi import APIRouter, Body, Depends, Form, HTTPException, UploadFile

from heavyiq.config import get_config
from heavyiq.langchain import HeavyDB
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.rag import models as md
from heavyrag.database import FactsModel, ragdb

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


# RAG database router
# which helps to make CRUD operations on RAG database, especially for facts
facts_db_router = APIRouter()


# insert fact
@facts_db_router.post("/insert")
async def add_fact(
    request: md.AddFactRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.AddFactResponse:
    """
    Endpoint for adding a new fact.
    """
    from heavyrag.ingest import ainsert_fact

    with ragdb.get_db() as session:
        fact_id = None
        try:
            # add db record
            fact_id = FactsModel.add(db_session=session, heavydb_name=heavydb._dbname, fact=request.fact)
            # add to index
            await ainsert_fact(fact_id=fact_id, fact=request.fact, heavydb_name=heavydb._dbname)

            return md.AddFactResponse(fact_id=fact_id)
        except Exception as e:
            if fact_id:
                FactsModel.delete(db_session=session, id=fact_id)
            raise e


# bulk insert facts
@facts_db_router.post("/bulk-insert")
async def bulk_insert_fact(
    request: md.BulkInsertFactsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.BulkInsertFactsResponse:
    """
    Bulk insert facts into sqlite database and store the relavant embeddings on vector database.
    """
    from heavyrag.ingest import ainsert_facts

    with ragdb.get_db() as session:
        session.begin()
        try:
            # add db record
            fact_ids = FactsModel.bulk_insert(db_session=session, heavydb_name=heavydb._dbname, facts=request.facts)
            # add to index
            await ainsert_facts(facts=list(zip(fact_ids, request.facts)), heavydb_name=heavydb._dbname)

            return md.BulkInsertFactsResponse(fact_ids=fact_ids)
        except Exception as e:
            session.rollback()
            raise e


# update fact
@facts_db_router.post("/update")
async def update_fact(
    request: md.UpdateFactRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.AddFactResponse:
    """
    Updates a particular fact.
    """
    from heavyrag.ingest import aupdate_fact

    with ragdb.get_db() as session:
        try:
            # add db record
            updated_id = FactsModel.update(db_session=session, id=request.fact_id, fact=request.fact)
            if not updated_id:
                raise ValueError(f"Failed to update fact id, {request.fact_id}")
            # add to index
            await aupdate_fact(fact_id=request.fact_id, fact=request.fact, heavydb_name=heavydb._dbname)

            return md.AddFactResponse(fact_id=request.fact_id)

        except Exception as e:
            raise e


@facts_db_router.post("/get")
async def get_fact(
    request: md.GetFactRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.FactResponse:
    """
    Return a particular fact by fact_id.
    """
    with ragdb.get_db() as session:
        fact = FactsModel.get(db_session=session, id=request.fact_id)
        if fact and fact.heavydb_name == heavydb._dbname:
            return md.FactResponse(fact=fact.fact)
        return md.FactResponse()


@facts_db_router.post("/delete")
async def delete_fact(
    request: md.DeleteFactRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.DeleteFactResponse:
    """
    Delete a particular database fact.
    """
    from heavyrag.ingest import adelete_facts

    with ragdb.get_db() as session:
        session.begin()
        try:
            deleted = FactsModel.delete(db_session=session, ids=request.fact_ids)
            if deleted:
                await adelete_facts(heavydb_name=heavydb._dbname, fact_ids=request.fact_ids)
                return md.DeleteFactResponse(deleted=True)
            return md.DeleteFactResponse(deleted=False)

        except Exception as e:
            session.rollback()
            raise e


@facts_db_router.post("/list")
async def list_facts(
    request: md.ListFactsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.ListFactsResponse:
    """
    List all facts associated with a database.
    """
    with ragdb.get_db() as session:
        facts = FactsModel.list(db_session=session, heavydb_name=heavydb._dbname, serialize=True)
        return md.ListFactsResponse(facts=facts)


@facts_db_router.post("/delete_all")
async def delete_facts(
    request: md.DeleteAllFactsRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> None:
    """
    Delete all facts asscoiated with a database. This involves deleteing database entries and index nodes.
    """
    from heavyrag.ingest import adelete_facts

    with ragdb.get_db() as session:
        await adelete_facts(heavydb_name=heavydb._dbname)
        FactsModel.delete_facts_by_database(db_session=session, heavydb_name=heavydb._dbname)
        return None


@facts_db_router.post("/index/nodes")
async def get_fact_nodes(
    request: md.GetFactsfromIndexRequest, heavydb: HeavyDB = Depends(get_current_heavydb_instance)
) -> md.GetFactsfromIndexResponse:
    """
    Get facts from vectorDB index.
    """
    from heavyrag.main import get_facts

    nodes = await get_facts(heavydb_name=heavydb._dbname, limit=request.limit)
    return md.GetFactsfromIndexResponse(
        nodes=[md.GetFactsfromIndexResponse.Node(content=i.get_text(), metadata=i.metadata) for i in nodes]
    )


@facts_db_router.post("/ask", response_model=md.AskFactsResponse)
async def ask_about_facts(
    request: md.AskFactsRequest, collection_name: str = Depends(get_collection_name)
) -> md.AskFactsResponse:
    """
    Ask questions regrading the uploaded facts.
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
        return md.AskFactsResponse(sources=out)

    response, eval_result = out
    passing, score = None, None
    if eval_result:
        passing = eval_result.passing
        score = eval_result.score

    return md.AskFactsResponse(
        answer=response.response,
        metadata=response.metadata,
        sources=[i.get_content(metadata_mode="all") for i in response.source_nodes],
        passing=passing,
        score=score,
    )
