from enum import Enum

from heavyiq.config import get_config

from .faiss import FaissIQVectorStore

CONFIG = get_config()


class VectorStoreType(Enum):
    """
    Type of the vectorstore.
    """

    CHROMA = "chroma"
    FAISS = "faiss"

    @classmethod
    def from_name(cls: type["VectorStoreType"], name: str) -> "VectorStoreType":
        """
        Select an Enum value based on the provided name (case-insensitive).

        Args:
            name (str): The name of the enum value.

        Returns:
            VectorStoreType: The corresponding Enum value.
        """
        for vtype in cls:
            if vtype.value.lower() == name.lower():
                return vtype
        raise ValueError(f"VectorStoreType with name {name} does not exist")


faiss_vs = None


def initialize_vector_db():
    """
    Helps to initialize vector db, loading index, etc.
    """
    global faiss_vs
    vector_type = VectorStoreType.from_name(CONFIG.rag_vectordb_type)
    if vector_type == VectorStoreType.FAISS:
        faiss_vs = FaissIQVectorStore(persist_dir=CONFIG.rag_faiss_persist_dir)
