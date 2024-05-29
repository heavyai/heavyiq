# Moule helps to transform list of documents into list of nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

# from heavyrag.index import embed_model

DEFAULT_PIPELINE = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)])


async def atransform(documents: list[Document], pipeline: IngestionPipeline | None = None) -> list[BaseNode]:
    """
    Helps to transform a list of documents to smaller nodes.
    """
    pipeline = pipeline or DEFAULT_PIPELINE
    splitted_nodes = await pipeline.arun(documents=documents, in_place=False, show_progress=True)

    return splitted_nodes
