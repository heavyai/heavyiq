# Moule helps to transform list of documents into list of nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, Document
from llama_index.core.text_splitter import TokenTextSplitter

# Token-based splitter respecting 512-token limit
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
