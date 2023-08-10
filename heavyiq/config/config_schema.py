from typing import Optional

from .overrides import OverrideBaseConfig


class CustomLLMConfig(OverrideBaseConfig):  # type: ignore
    api_base: str
    context_window: int


class DevConfig(OverrideBaseConfig):  # type: ignore
    nl_to_sql: CustomLLMConfig
    sql_to_answer: CustomLLMConfig


class HeavyIQConfig(OverrideBaseConfig):  # type: ignore
    port: int = 6275
    data: Optional[str] = None
    openai_api_key: str
    openai_gpt_model: str = "gpt-3.5-turbo-16k"
    heavydb_dbname: Optional[str] = None
    heavydb_username: Optional[str] = None
    heavydb_password: Optional[str] = None
    heavydb_host: str = "localhost"
    heavydb_port: int = 6278
    heavydb_protocol: str = "binary"
    huggingface_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    metadata_index_dir: str = "db"
    table_documents_dir: str = "table_documents"
    # TELEMETRICS
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
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
    dev: Optional[DevConfig] = None


class AppConfig(OverrideBaseConfig):  # type: ignore
    data: Optional[str] = None
    iq: HeavyIQConfig
