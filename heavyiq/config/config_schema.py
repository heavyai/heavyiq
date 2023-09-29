from typing import Optional

from .overrides import OverrideBaseConfig


class HeavyIQConfig(OverrideBaseConfig):  # type: ignore
    port: int = 6275
    data: Optional[str] = None
    openai_api_key: str
    openai_gpt_model: str = "gpt-3.5-turbo-16k"  # LLMType.DEFAULT
    openai_gpt_model_nl_to_sql: Optional[str] = None  # LLMType.NL_TO_SQL
    openai_gpt_model_sql_to_answer: Optional[str] = None  # LLMType.SQL_TO_ANSWER
    max_fields_limit_sql_to_answer: int = 20  # LLMType.SQL_TO_ANSWER
    """Max number of fields (rows x columns) allowed to pass onto sql_to_answer llm"""
    heavydb_dbname: Optional[str] = None
    heavydb_username: Optional[str] = None
    heavydb_password: Optional[str] = None
    heavydb_host: str = "localhost"
    heavydb_port: int = 6278
    heavydb_protocol: str = "binary"
    huggingface_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    metadata_index_dir: str = "db"
    docs_index_dir: str = "heavyai_docs"
    table_documents_dir: str = "table_documents"
    enable_str_literal_correction: bool = True
    column_top_k_cardinality_threshold: int = 5
    """If the cardinality of a column is less than this threshold, we will include each value of the column in the prompt"""
    column_top_k_high_cardinality_sample: int = 3
    """If the cardinality of a column is greater than the threshold, we will sample this many values from the column"""
    # TELEMETRICS
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    # LOGGING
    access_log_level: str = "INFO"
    """Used for the access log of the web server."""
    heavyiq_log_level: str = "INFO"
    """Used for the log of the application code."""
    log_to_stdout: bool = False
    enable_debug_endpoints: bool = False
    # CUSTOM LLM
    custom_llm_type: Optional[str] = None
    """ 'API' or 'API_VLLM' or 'AZURE' """
    custom_llm_api_base: str = ""  # LLMType.DEFAULT
    custom_llm_api_context_window: int = 2048  # LLMType.DEFAULT
    custom_llm_api_nl_to_sql_base: Optional[str] = None  # LLMType.NL_TO_SQL
    custom_llm_api_nl_to_sql_context_window: int = 2048  # LLMType.NL_TO_SQL
    custom_llm_api_sql_to_answer_base: Optional[str] = None  # LLMType.SQL_TO_ANSWER
    custom_llm_api_sql_to_answer_context_window: int = 2048  # LLMType.SQL_TO_ANSWER
    custom_llm_api_vllm_beam_width: int = 1
    custom_llm_azure_openai_api_version: str = "2023-03-15-preview"
    custom_llm_azure_openai_api_base: str = ""
    custom_llm_azure_deployment_name: str = ""


class AppConfig(OverrideBaseConfig):  # type: ignore
    data: Optional[str] = None
    iq: HeavyIQConfig
