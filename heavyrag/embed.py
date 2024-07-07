import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from heavyiq.config import get_config
from heavyrag.logger import logger

CONFIG, EMBED_MODEL, CHROMA_CLIENT = get_config(), None, None


def set_embed_model():
    """
    Supposed to set the embed model
    """
    global EMBED_MODEL

    if CONFIG.rag_embed_server_base:
        logger.info(
            "Initialized embed model in GPU bound inference using TextEmbeddingsInference with the following parameters: "
            f"model_name={CONFIG.rag_embed_model_name}, timeout=60 seconds, embed_batch_size=10, base_url={CONFIG.rag_embed_server_base}"
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


def get_embed_model():
    """
    Get the embed model.
    """
    if EMBED_MODEL is None:
        set_embed_model()
    return EMBED_MODEL


def set_chroma_client():
    global CHROMA_CLIENT
    CHROMA_CLIENT = chromadb.PersistentClient(
        path=CONFIG.rag_chromadb_persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True),
    )


def get_chroma_client():
    if CHROMA_CLIENT is None:
        set_chroma_client()
    return CHROMA_CLIENT
