from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from heavyrag.documents.enums import DocType
from heavyrag.settings import settings


def get_pipeline_by_document_type(doctype: DocType):
    """
    Gets the pipeline for transformations by document type.
    """
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=500, chunk_overlap=20, paragraph_separator="\n\n"),
            HuggingFaceEmbedding(model_name=settings.hf_embedding_model),
        ],
    )
    if doctype == DocType.TXT:
        return pipeline
    elif doctype == DocType.PDF:
        return pipeline
    return pipeline
