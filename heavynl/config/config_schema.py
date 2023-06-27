from typing import Optional

from pydantic import Field

from .overrides import OverrideConfZ


class LogConfig(OverrideConfZ):  # type: ignore
    app_log_file: str = "access.log"
    app_log_level: str = "INFO"
    heavyiq_log_file: str = "heavyiq.log"
    heavyiq_log_level: str = "INFO"


class HeavyNLConfig(OverrideConfZ):  # type: ignore
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
    log: LogConfig = LogConfig()


class HeavyWebConfig(OverrideConfZ):  # type: ignore
    backend_url: Optional[str] = Field(alias="backend-url")


class AppConfig(OverrideConfZ):  # type: ignore
    http_port: Optional[int] = Field(alias="http-port")
    nl: HeavyNLConfig
    web: Optional[HeavyWebConfig] = None
