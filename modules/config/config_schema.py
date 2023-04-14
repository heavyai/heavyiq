from confz import ConfZ, ConfZFileSource


class HeavyNL(ConfZ):  # type: ignore
    heavydb_host: str
    heavydb_port: int
    heavydb_dbname: str
    heavydb_username: str
    heavydb_password: str
    openai_api_key: str
    huggingface_embed_model: str


class AppConfig(ConfZ):  # type: ignore
    heavynl: HeavyNL

    CONFIG_SOURCES = [ConfZFileSource(file="./config.toml")]
