from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """
    Uvicorn settings
    """

    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8006, description="Listen port")
    app_name: str = Field(
        default="HeavyIQ RAG", description="RAG approach using llama-index"
    )
    debug: bool = Field(default=False, description="Run on debug mode or not.")


class RAGSettings(BaseSettings):
    """
    Settings related to RAG.
    """

    persistant_storage_dir: str = Field(
        default="index-storage",
        description="Local directory for storage index related data.",
    )
    persistant_collection_dir: str = Field(
        default="chromadb", description="Local directory where chormadb resides."
    )
    hf_embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5", description="HuggingFace Embedding model name."
    )
    collection_name: str = Field(
        default="iqrag", description="Name of the chromadb collection."
    )


class HeavyDBSettings(BaseSettings):
    """
    Settings related to HeavyDB connection establishment.
    """

    heavydb_host: str = Field(default="localhost", description="HeavyDB Host")
    heavydb_port: int = Field(default=6278, description="HeavyDB Port")
    heavydb_protocol: str = Field(default="binary", description="HeavyDB protocol")


class Settings(ServerSettings, RAGSettings, HeavyDBSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
