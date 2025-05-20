# This module should be loaded on gunicorn worker process
# Moule helps to transform list of documents into list of nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, Document
from llama_index.core.text_splitter import TokenTextSplitter

from heavyrag.embed import EMBED_MODEL_MAX_LEN

# EMBED_MODEL_MAX_LEN variable being set on master process and then made available to all worker processes via --preload option
# this option copies all the global varibales to the forked process
if EMBED_MODEL_MAX_LEN:
    token_splitter = TokenTextSplitter(chunk_size=EMBED_MODEL_MAX_LEN, chunk_overlap=20)
else:
    token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=20)

DEFAULT_PIPELINE = IngestionPipeline(
    transformations=[token_splitter]  # no need for SentenceSplitter if token chunking suffices
)


async def atransform(documents: list[Document], pipeline: IngestionPipeline | None = None) -> list[BaseNode]:
    """
    Helps to transform a list of documents to smaller nodes.
    """
    pipeline = pipeline or DEFAULT_PIPELINE
    splitted_nodes = await pipeline.arun(documents=documents, in_place=False, show_progress=True)

    return splitted_nodes
