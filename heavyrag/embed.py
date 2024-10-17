# import chromadb
# from chromadb.api import ClientAPI
# from chromadb.config import Settings as ChromaDBSettings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from heavyiq.config import get_config
from heavyiq.utils import get_host_and_port
from heavyrag.logger import logger

CONFIG, EMBED_MODEL = get_config(), None


def set_embed_model():
    """
    Supposed to set the embed model
    """
    from heavyiq.langchain.llms import get_vllm_model_name

    global EMBED_MODEL
    if not CONFIG.enable_rag:
        msg = "Failed to initialize the embedding model. The enable_rag setting is being turned off."
        logger.error(msg)
        raise ValueError(msg)

    if CONFIG.rag_embed_server_base and CONFIG.rag_embed_server_base.endswith("/v1"):
        model_name = get_vllm_model_name(CONFIG.rag_embed_server_base)
        EMBED_MODEL = OpenAIEmbedding(
            api_key="nothing",
            api_base=CONFIG.rag_embed_server_base,
            model_name=model_name,
            embed_batch_size=100,
        )
        logger.info(
            "Initialized embed model using OpenAIEmbedding client with the following parameters: "
            f"timeout=60 seconds, embed_batch_size=100, model_name={model_name}, base_url={CONFIG.rag_embed_server_base}"
        )
    elif CONFIG.rag_embed_server_base:
        logger.info(
            "Initialized embed model in GPU bound inference using TextEmbeddingsInference client with the following parameters: "
            f"timeout=60 seconds, embed_batch_size=10, base_url={CONFIG.rag_embed_server_base}"
        )
        EMBED_MODEL = TextEmbeddingsInference(
            model_name=CONFIG.rag_embed_model_name,  # required for formatting inference text,
            timeout=60,  # timeout in seconds
            embed_batch_size=10,  # batch size for embeddin
            base_url=CONFIG.rag_embed_server_base,
        )
    else:
        # logger.info(
        #    f"Initialized embed model on CPU using HuggingFaceEmbedding with model name: {CONFIG.rag_embed_model_name}"
        # )
        # EMBED_MODEL = HuggingFaceEmbedding(CONFIG.rag_embed_model_name, device="cpu") # this fails to generate embeddings
        EMBED_MODEL = None


def get_embed_model():
    """
    Get the embed model.
    """
    if EMBED_MODEL is None:
        set_embed_model()
    return EMBED_MODEL
