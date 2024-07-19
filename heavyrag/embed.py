import chromadb
from chromadb.api import ClientAPI
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from heavyiq.config import get_config
from heavyrag.logger import logger

CONFIG, EMBED_MODEL, CHROMA_CLIENT = get_config(), None, None


class LlamaIndexEmbeddingAdapter(chromadb.EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [node.embedding for node in self.ef([TextNode(text=doc) for doc in input])]


def set_embed_model():
    """
    Supposed to set the embed model
    """
    global EMBED_MODEL

    if CONFIG.rag_embed_server_base:
        logger.info(
            "Initialized embed model in GPU bound inference using TextEmbeddingsInference with the following parameters: "
            f"timeout=60 seconds, embed_batch_size=10, base_url={CONFIG.rag_embed_server_base}"
        )
        EMBED_MODEL = TextEmbeddingsInference(
            model_name=CONFIG.rag_embed_model_name,  # required for formatting inference text,
            timeout=60,  # timeout in seconds
            embed_batch_size=10,  # batch size for embeddin
            base_url=CONFIG.rag_embed_server_base,
        )
    else:
        logger.info(
            f"Initialized embed model on CPU using HuggingFaceEmbedding with model name: {CONFIG.rag_embed_model_name}"
        )
        EMBED_MODEL = HuggingFaceEmbedding(CONFIG.rag_embed_model_name, device="cpu")


def get_embed_model() -> BaseEmbedding:
    """
    Get the embed model.
    """
    if EMBED_MODEL is None:
        set_embed_model()
    return EMBED_MODEL


def get_chroma_client() -> ClientAPI:
    return chromadb.PersistentClient(
        path=CONFIG.rag_chromadb_persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True),
    )
