import os
from urllib.parse import urlparse

from confz import FileSource
from heavydb.thrift.Heavy import Client
from heavydb.thrift.ttypes import TLicenseInfo
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport
import openai


from .config_schema import AppConfig, HeavyNLConfig

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


def get_heavydb_license_claims(config: HeavyNLConfig) -> TLicenseInfo:
    try:
        socket = TSocket.TSocket(config.heavydb_host, config.heavydb_port)
        socket.setTimeout(8000)  # try to connect for 8 seconds
        transport = TTransport.TBufferedTransport(socket)
        proto = TBinaryProtocol.TBinaryProtocolAccelerated(transport)
        transport.open()
        client = Client(proto)
        res = client.get_license_claims(None, "")
        transport.close()
        return res
    except Exception as e:
        raise ValueError(
            f"Unable to communicate with HeavyDB instance at {config.heavydb_host}:{config.heavydb_port}: {e}"
        )


def get_config(file: str = "./config.toml") -> HeavyNLConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config
    if _config:
        return _config
    app_config = AppConfig(config_sources=FileSource(file=file))
    if app_config.iq.custom_llm_type is None or app_config.iq.custom_llm_type == "AZURE":
        if app_config.iq.custom_llm_type == "AZURE" and (
            app_config.iq.custom_llm_azure_deployment_name.strip() == ""
            or app_config.iq.custom_llm_azure_openai_api_base.strip() == ""
            or app_config.iq.custom_llm_azure_openai_api_version.strip() == ""
        ):
            raise ValueError(
                "Custom LLM type is set to 'AZURE', but deployment name or API base URL or API Version is not set."
            )
        openai.api_key = app_config.iq.openai_api_key
        if app_config.iq.custom_llm_type is None:
            try:
                openai.Model.list()
            except Exception as e:
                raise ValueError(f"Unable to communicate with OpenAI API: {e}")
    elif app_config.iq.custom_llm_type == "API":
        if app_config.iq.custom_llm_api_base.strip() == "":
            raise ValueError("Custom LLM type is set to 'API', but API base URL is not set.")
    else:
        raise ValueError(f"Invalid custom LLM type (valid options are AZURE or API): {app_config.iq.custom_llm_type}")
    if app_config.web and app_config.web.backend_url:
        hostname, port = split_url_port(app_config.web.backend_url)
        app_config.iq.heavydb_host = hostname
        app_config.iq.heavydb_port = port
    elif app_config.http_port:
        app_config.iq.heavydb_port = app_config.http_port
    if app_config.iq.data is None:
        if app_config.web and app_config.web.data:
            app_config.iq.data = app_config.web.data
        else:
            app_config.iq.data = "./storage"
    if not os.path.exists(app_config.iq.data):
        os.makedirs(app_config.iq.data)
    _config = app_config.iq

    # disabled for now because HeavyIQ can startup before HeavyDB
    # get_heavydb_license_claims(_config)

    return _config
