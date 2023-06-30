from typing import Optional

from pydantic import Field

from .overrides import OverrideConfZ


class LogConfig(OverrideConfZ):  # type: ignore
    app_log_file: str = "access.log"
    app_log_level: str = "INFO"
    heavyiq_log_file: str = "heavyiq.log"
    heavyiq_log_level: str = "INFO"


class CustomLLMConfig(OverrideConfZ):  # type: ignore
    type: str
    """ 'API' or 'LOCAL' or 'AZURE' """
    api_base: str = ""
    api_context_window: int = 2048
    openai_api_version: str = "2023-03-15-preview"
    openai_api_base: str = ""
    azure_deployment_name: str = ""


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
    custom_llm: Optional[CustomLLMConfig] = None


class HeavyWebConfig(OverrideConfZ):  # type: ignore
    backend_url: Optional[str] = Field(alias="backend-url")


class AppConfig(OverrideConfZ):  # type: ignore
    http_port: Optional[int] = Field(alias="http-port")
    iq: HeavyNLConfig
    web: Optional[HeavyWebConfig] = None
