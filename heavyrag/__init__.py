from .embed import set_embed_model
from .vector_stores.base import initialize_vector_db


class IndexNotFound(Exception):
    pass


def initialize_rag():
    """
    Initialize models and clients (ie. HF models and chroma clients) relevant to heavyrag.
    """
    set_embed_model()
    # initialize vector stores
    initialize_vector_db()
