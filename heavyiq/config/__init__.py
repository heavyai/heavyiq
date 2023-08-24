import os
from urllib.parse import urlparse

from confz import FileSource
from heavydb.thrift.Heavy import Client  # type: ignore
from heavydb.thrift.ttypes import TLicenseInfo
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport
import openai


from .config_schema import AppConfig, HeavyIQConfig

_config: HeavyIQConfig | None = None


def get_heavydb_license_claims(config: HeavyIQConfig) -> TLicenseInfo:
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


def get_config(file: str = "./config.toml") -> HeavyIQConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config
    if _config:
        return _config
    app_config: AppConfig = AppConfig(config_sources=FileSource(file=file))  # type: ignore
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
    elif app_config.iq.custom_llm_type == "API_VLLM":
        if app_config.iq.custom_llm_api_base.strip() == "":
            raise ValueError("Custom LLM type is set to 'API_VLLM', but API base URL is not set.")
        if app_config.iq.custom_llm_api_vllm_model_name is None:
            raise ValueError("Custom LLM type is set to 'API_VLLM', but API VLLM model name is not set.")
    else:
        raise ValueError(f"Invalid custom LLM type (valid options are AZURE or API): {app_config.iq.custom_llm_type}")
    if app_config.iq.data is None:
        if app_config.data:
            app_config.iq.data = app_config.data
        else:
            app_config.iq.data = "./storage"
    if not os.path.exists(app_config.iq.data):
        os.makedirs(app_config.iq.data)
    _config = app_config.iq

    # disabled for now because HeavyIQ can startup before HeavyDB
    # get_heavydb_license_claims(_config)

    return _config
