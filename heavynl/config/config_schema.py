from typing import Optional

from confz import ConfZ
from pydantic import Field


class HeavyNLConfig(ConfZ):  # type: ignore
    port: int = 6275
    openai_api_key: str
    openai_gpt_model: str = "gpt-3.5-turbo"
    heavydb_dbname: Optional[str] = None
    heavydb_username: Optional[str] = None
    heavydb_password: Optional[str] = None
    heavydb_host: str = "localhost"
    heavydb_port: int = 6278
    huggingface_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    metadata_index_dir: str = "db"
    table_documents_dir: str = "table_documents"
    promptlayer_api_key: Optional[str] = None
    promptwatch_api_key: Optional[str] = None
    promptwatch_tracking_project: Optional[str] = None


class HeavyWebConfig(ConfZ):  # type: ignore
    backend_url: Optional[str] = Field(alias="backend-url")


class LogConfig(ConfZ):  # type: ignore
    app_log_file: Optional[str] = None
    app_log_level: Optional[str] = None
    heavynl_log_file: Optional[str] = None
    heavynl_log_level: Optional[str] = None


class AppConfig(ConfZ):  # type: ignore
    http_port: Optional[int] = Field(alias="http-port")
    nl: HeavyNLConfig
    web: Optional[HeavyWebConfig] = None
    log: LogConfig
