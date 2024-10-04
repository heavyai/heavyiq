class IndexNotFound(Exception):
    pass


def initialize_rag():
    """
    Initialize models and clients (ie. HF models and chroma clients) relevant to heavyrag.
    """
    from .embed import set_embed_model

    set_embed_model()
