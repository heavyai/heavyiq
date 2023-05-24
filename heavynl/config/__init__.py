from confz import ConfZFileSource

from .config_schema import AppConfig, HeavyNLConfig

_config = None


def get_config(file: str = "./config.toml") -> HeavyNLConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config
    if _config:
        return _config
    _config = AppConfig(config_sources=ConfZFileSource(file=file)).nl
    return _config
