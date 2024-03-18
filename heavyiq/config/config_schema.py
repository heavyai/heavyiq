from typing import Optional

from .overrides import OverrideBaseConfig


class HeavyIQConfig(OverrideBaseConfig):  # type: ignore
    port: int = 6275
    data: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_gpt_model: str = "gpt-3.5-turbo-16k"  # LLMType.DEFAULT
    openai_gpt_model_nl_to_sql: Optional[str] = None  # LLMType.NL_TO_SQL
    openai_gpt_model_sql_to_answer: Optional[str] = None  # LLMType.SQL_TO_ANSWER
    openai_gpt_model_nl_to_tables: Optional[str] = None  # LLMType.NL_TO_TABLES
    openai_gpt_model_instruct: Optional[str] = None  # LLMType.INSTRUCT
    openai_gpt_model_tables_to_questions: Optional[str] = None  # LLMType.TABLES_TO_QUESTIONS
    max_fields_limit_sql_to_answer: int = 20  # LLMType.SQL_TO_ANSWER
    """Max number of fields (rows x columns) allowed to pass onto sql_to_answer llm"""
    heavydb_dbname: Optional[str] = None
    heavydb_username: Optional[str] = None
    heavydb_password: Optional[str] = None
    heavydb_host: str = "localhost"
    heavydb_port: int = 6278
    heavydb_protocol: str = "binary"
    huggingface_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    huggingface_model_cache_folder: str = "hf_models"
    huggingface_embed_documents_parallel: bool = False
    """If True, then it uses multiple processes for embedding documents"""
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
    access_log_max_file_size: int = 104857600
    """Default Access logs max file size is 100 MB"""
    heavyiq_log_max_file_size: int = 104857600
    """Default Application's log max file size is 100 MB"""
    log_to_stdout: bool = False
    enable_debug_endpoints: bool = False
    enable_llm_cache: bool = False
    # CUSTOM LLM
    custom_llm_type: Optional[str] = None
    """ 'API' or 'API_VLLM' or 'AZURE' """
    custom_llm_api_base: str = ""  # LLMType.DEFAULT
    custom_llm_api_context_window: int = 8192  # LLMType.DEFAULT
    custom_llm_api_nl_to_sql_base: Optional[str] = None  # LLMType.NL_TO_SQL
    custom_llm_api_nl_to_sql_context_window: int = 8192  # LLMType.NL_TO_SQL
    custom_llm_api_sql_to_answer_base: Optional[str] = None  # LLMType.SQL_TO_ANSWER
    custom_llm_api_sql_to_answer_context_window: int = 8192  # LLMType.SQL_TO_ANSWER
    custom_llm_api_nl_to_tables_base: Optional[str] = None  # LLMType.NL_TO_TABLES
    custom_llm_api_nl_to_tables_context_window: int = 8192  # LLMType.NL_TO_TABLES
    custom_llm_api_tables_to_questions_base: Optional[str] = None  # LLMType.TABLES_TO_QUESTIONS
    custom_llm_api_tables_to_questions_context_window: int = 8192  # LLMType.TABLES_TO_QUESTIONS
    custom_llm_api_instruct_base: Optional[str] = None  # LLMType.INSTRUCT
    custom_llm_api_instruct_context_window: int = 8192  # LLMType.INSTRUCT
    custom_llm_api_instruct_prompt_start_token: str = "<|prompt|>\n"
    custom_llm_api_instruct_prompt_end_token: str = "\n<|answer|>\n"
    custom_llm_api_tables_to_questions_prompt_token: str = "<|table question prompt|>"
    custom_llm_api_tables_to_questions_answer_token: str = "<|table question answer|>"
    custom_llm_api_vllm_beam_width: int = 2
    custom_llm_api_vllm_max_tokens: int = 512
    custom_llm_azure_openai_api_version: str = "2023-03-15-preview"
    custom_llm_azure_openai_api_base: str = ""
    custom_llm_azure_deployment_name: str = ""
    # logprobs
    enable_logprobs: bool = False
    custom_llm_logprobs_limit: int = 1
    # top-k
    top_k_max_str_col_count_nl_to_sql: int = 30
    top_k_max_str_col_count_nl_to_tables: int = 30
    # table order in prompt
    sort_prompt_tables_desc: bool = True
    # whether to include timestamp text on table prompt
    include_timestamp_on_table_info_prompt: bool = True
    # form a inline-column metadata(top-k, time-range) on table_info generation part of prompt making
    # if False, then column top-k and time-ranges should appear in a seperate paragraph block
    inline_column_metadata_on_table_info_prompt: bool = False
    # maximum count of allowed_tables on nl-to-tables endpoint. If the count exceeds, then
    # index updation and search will happen in-order to find relevant tables.
    allowed_tables_max_count_nl_to_tables: int = 20
    # config options w.r.t table doc generation using custom model
    include_samples_in_custom_table_document_generation: bool = False
    include_top_k_in_custom_table_document_generation: bool = True
    # nl to sql max query retries
    max_retries_nl_to_sql: int = 2
    # llm api-key
    heavylm_api_key: Optional[str] = None


class AppConfig(OverrideBaseConfig):  # type: ignore
    data: Optional[str] = None
    iq: HeavyIQConfig
