from confz import ConfZ, ConfZFileSource


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


class AppConfig(ConfZ):  # type: ignore
    nl: HeavyNLConfig

    CONFIG_SOURCES = [ConfZFileSource(file="./config.toml")]
