from .embed import set_chroma_client, set_embed_model


class IndexNotFound(Exception):
    pass


def initialize_rag():
    """
    Initialize models and clients (ie. HF models and chorma clients) relevant to heavyrag.
    """
    set_embed_model()
    set_chroma_client()
