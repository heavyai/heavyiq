from typing import Optional

from pydantic import Field

from .overrides import OverrideBaseConfig


class HeavyNLConfig(OverrideBaseConfig):  # type: ignore
    port: int = 6275
    data: Optional[str] = None
    openai_api_key: str
    openai_gpt_model: str = "gpt-3.5-turbo-16k"
    heavydb_dbname: Optional[str] = None
    heavydb_username: Optional[str] = None
    heavydb_password: Optional[str] = None
    heavydb_host: str = "localhost"
    heavydb_port: int = 6274
    heavydb_protocol: str = "binary"
    huggingface_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    metadata_index_dir: str = "db"
    table_documents_dir: str = "table_documents"
    # TELEMETRICS
    promptlayer_api_key: Optional[str] = None
    promptwatch_api_key: Optional[str] = None
    promptwatch_tracking_project: Optional[str] = None
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = None
    # LOGGING
    access_log_level: str = "INFO"
    """Used for the access log of the web server."""
    heavyiq_log_level: str = "INFO"
    """Used for the log of the application code."""
    # CUSTOM LLM
    custom_llm_type: Optional[str] = None
    """ 'API' or 'AZURE' """
    custom_llm_api_base: str = ""
    custom_llm_api_context_window: int = 2048
    custom_llm_azure_openai_api_version: str = "2023-03-15-preview"
    custom_llm_azure_openai_api_base: str = ""
    custom_llm_azure_deployment_name: str = ""


class AppConfig(OverrideBaseConfig):  # type: ignore
    data: Optional[str] = None
    iq: HeavyNLConfig
