# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaDBSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from heavyiq.config import get_config
from heavyiq.utils import get_host_and_port
from heavyrag.embed import get_embed_model

CONFIG, CHROMA_CLIENT = get_config(), None


class LlamaIndexEmbeddingAdapter(chromadb.EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [node.embedding for node in self.ef([TextNode(text=doc) for doc in input])]


def set_chroma_client():
    global CHROMA_CLIENT
    try:
        host, port = get_host_and_port(CONFIG.rag_chromadb_server_base)
        CHROMA_CLIENT = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaDBSettings(anonymized_telemetry=False),
        )

    except Exception as e:
        print(f"Failed to create ChromaDB HTTP Client, {e}")
        CHROMA_CLIENT = None


def get_chroma_client() -> ClientAPI | None:
    if not get_embed_model():
        return None
    if CHROMA_CLIENT is None:
        set_chroma_client()
    return CHROMA_CLIENT


class ChromaIQVectorStore(ChromaVectorStore):
    """
    Attributes which are specific to the chromaDB vectorstore are available here.
    """

    def __init__(
        self, collection_name: str, metadata: dict | None = None, create_collection_if_not_exists: bool = True
    ):
        client = get_chroma_client()
        embedding_func = LlamaIndexEmbeddingAdapter(get_embed_model())
        if create_collection_if_not_exists:
            metadata = metadata or {}
            metadata.update({"hnsw:space": "cosine"})
            collection = client.get_or_create_collection(
                collection_name, metadata=metadata, embedding_function=embedding_func
            )
        else:
            try:
                collection = client.get_collection(collection_name, embedding_function=embedding_func)
            except Exception:
                raise ValueError(f"Collection {collection_name} not found!")
        super().__init__(chroma_collection=collection)

    @classmethod
    def class_name(cls: type["ChromaIQVectorStore"]) -> str:
        return "ChromaIQVectorStore"
