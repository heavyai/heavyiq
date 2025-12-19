# This module should be loaded on gunicorn worker process
# Moule helps to transform list of documents into list of nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, Document
from llama_index.core.text_splitter import TokenTextSplitter

from heavyiq.langchain.utils import custom_model_tokenizer_encode
from heavyrag.embed import EMBED_MODEL_MAX_LEN

# EMBED_MODEL_MAX_LEN variable being set on master process and then made available to all worker processes via --preload option
# this option copies all the global variables to the forked process
#
# Buffer of 50 tokens is needed because LlamaIndex adds metadata (extra_info, metadata)
# when building the final chunk string, and this contributes to the total token count.
METADATA_BUFFER = 50

if EMBED_MODEL_MAX_LEN:
    chunk_size = EMBED_MODEL_MAX_LEN - METADATA_BUFFER
else:
    # Fallback for edge cases - use conservative default (512 - buffer)
    chunk_size = 512 - METADATA_BUFFER

token_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=20, tokenizer=custom_model_tokenizer_encode)

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
