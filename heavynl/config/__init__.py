from urllib.parse import urlparse

from confz import ConfZFileSource
import openai

from .config_schema import AppConfig, HeavyNLConfig, LogConfig

_config = None


def split_url_port(url: str) -> tuple[str, int]:
    """
    Parse the given URL and extract hostname and port.

    Args:
        url (str): The URL to be parsed.

    Returns:
        tuple: A tuple containing the hostname and port as string and integer respectively.
    """
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        raise ValueError(f"Invalid URL contains either no hostname or port: {url}")

    return parsed.hostname, parsed.port


def get_config(file: str = "./config.toml") -> HeavyNLConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config
    if _config:
        return _config
    app_config = AppConfig(config_sources=ConfZFileSource(file=file))
    openai.api_key = app_config.nl.openai_api_key
    try:
        openai.Model.list()
    except Exception as e:
        raise ValueError(f"Unable to communicate with OpenAI: {e}")
    if app_config.web and app_config.web.backend_url:
        hostname, port = split_url_port(app_config.web.backend_url)
        app_config.nl.heavydb_host = hostname
        app_config.nl.heavydb_port = port
    elif app_config.http_port:
        app_config.nl.heavydb_port = app_config.http_port
    _config = app_config.nl
    return _config


def get_log_config(file: str = "./config.toml") -> LogConfig:
    """
    Get config related to logging.
    """
    return get_config(file).log
