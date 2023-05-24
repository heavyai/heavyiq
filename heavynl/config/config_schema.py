from typing import Optional

from confz import ConfZ


class HeavyNLConfig(ConfZ):  # type: ignore
    heavydb_host: str
    heavydb_port: int
    heavydb_dbname: str
    heavydb_username: str
    heavydb_password: str
    openai_api_key: str
    huggingface_embed_model: str
    metadata_index_dir: str
    table_documents_dir: str
    promptlayer_api_key: Optional[str] = None
    promptwatch_api_key: Optional[str] = None
    promptwatch_tracking_project: Optional[str] = None


class AppConfig(ConfZ):  # type: ignore
    nl: HeavyNLConfig
