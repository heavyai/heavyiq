# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# import chromadb
# from chromadb.api import ClientAPI
# from chromadb.config import Settings as ChromaDBSettings
from concurrent.futures import ThreadPoolExecutor

from llama_index.embeddings.openai import OpenAIEmbedding

from heavyiq.config import get_config
from heavyrag.logger import logger

CONFIG, EMBED_MODEL, EMBED_MODEL_MAX_LEN = get_config(), None, None


def get_embed_model_info() -> tuple[str, int]:
    """
    Get info regrading the embedding model.
    """
    from heavyiq.langchain.llms import get_vllm_max_model_len, get_vllm_model_name

    model_name = get_vllm_model_name(CONFIG.rag_embed_server_base)
    max_model_len = get_vllm_max_model_len(CONFIG.rag_embed_server_base)

    return model_name, max_model_len


def set_embed_model():
    """
    Supposed to set the embed model
    """

    global EMBED_MODEL, EMBED_MODEL_MAX_LEN
    if not CONFIG.enable_rag:
        msg = "Failed to initialize the embedding model. The enable_rag setting is being turned off."
        logger.error(msg)
        raise ValueError(msg)

    if CONFIG.rag_embed_server_base and CONFIG.rag_embed_server_base.endswith("/v1"):
        model_name, model_max_len = get_embed_model_info()
        EMBED_MODEL = OpenAIEmbedding(
            api_key="nothing",
            api_base=CONFIG.rag_embed_server_base,
            model_name=model_name,
            embed_batch_size=100,
        )
        EMBED_MODEL_MAX_LEN = model_max_len
        logger.info(
            "Initialized embed model using OpenAIEmbedding client with the following parameters: "
            f"timeout=60 seconds, embed_batch_size=100, model_name={model_name}, base_url={CONFIG.rag_embed_server_base}"
        )
    elif CONFIG.rag_embed_server_base:
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

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
        EMBED_MODEL = None


def get_embed_model():
    """
    Get the embed model.
    """
    if EMBED_MODEL is None:
        set_embed_model()
    return EMBED_MODEL
