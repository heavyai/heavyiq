import os
import threading
from typing import Any
from urllib.parse import urlparse

from confz import FileSource
from heavydb.thrift.Heavy import Client  # type: ignore
from heavydb.thrift.ttypes import TLicenseInfo
from openai import OpenAI
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport

from .config_schema import AppConfig, HeavyIQConfig

_config: HeavyIQConfig | None = None
_config_lock = threading.Lock()


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


def validate_config(app_config: AppConfig) -> AppConfig:
    """
    Validates app config after it's creation.
    This involves basic validation of configs and creation of directory mentioned if not exists.
    """
    if app_config.iq.custom_llm_type is None or app_config.iq.custom_llm_type == "AZURE":
        if app_config.iq.custom_llm_type == "AZURE" and (
            app_config.iq.custom_llm_azure_deployment_name.strip() == ""
            or app_config.iq.custom_llm_azure_openai_api_base.strip() == ""
            or app_config.iq.custom_llm_azure_openai_api_version.strip() == ""
        ):
            raise ValueError(
                "Custom LLM type is set to 'AZURE', but deployment name or API base URL or API Version is not set."
            )
        openai_client = OpenAI(api_key=app_config.iq.openai_api_key)
        if app_config.iq.custom_llm_type is None:
            try:
                openai_client.models.list()
            except Exception as e:
                raise ValueError(f"Unable to communicate with OpenAI API: {e}")
    elif app_config.iq.custom_llm_type == "API":
        if app_config.iq.custom_llm_api_base.strip() == "":
            raise ValueError("Custom LLM type is set to 'API', but API base URL is not set.")
    elif app_config.iq.custom_llm_type == "API_VLLM":
        if app_config.iq.custom_llm_api_base.strip() == "":
            raise ValueError("Custom LLM type is set to 'API_VLLM', but API base URL is not set.")
    else:
        raise ValueError(
            f"Invalid custom LLM type (valid options are AZURE or API or API_VLLM): {app_config.iq.custom_llm_type}"
        )
    if app_config.iq.data is None:
        if app_config.data:
            app_config.iq.data = app_config.data
        else:
            app_config.iq.data = "./storage"
    if not os.path.exists(app_config.iq.data):
        os.makedirs(app_config.iq.data)

    return app_config


def get_config(file: str = "./config.toml") -> HeavyIQConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config
    with _config_lock:
        if _config:
            return _config
        app_config: AppConfig = AppConfig(config_sources=FileSource(file=file))  # type: ignore
        app_config = validate_config(app_config)
        _config = app_config.iq

        return _config


def change_iq_config(dict_: dict[str, Any]) -> bool:
    """
    Helps to change App.iq config w.r.t passed dict.
    """
    global _config
    if not _config or not dict_:
        return False

    with _config_lock:
        for key, value in dict_.items():
            setattr(_config, key, value)

    return True


def change_iq_config_for_free_edition() -> bool:
    """
    IQ config keys to change if in case of free edition license.
    """

    keys_to_change = {
        "custom_llm_api_base": "https://api2.heavy.ai/v1",
        "custom_llm_api_nl_to_sql_base": "https://api2.heavy.ai/v1",
        "custom_llm_api_sql_to_answer_base": "https://api2.heavy.ai/v1",
        "custom_llm_api_nl_to_tables_base": "https://api2.heavy.ai/v1",
        "custom_llm_api_tables_to_questions_base": "https://api2.heavy.ai/v1",
        "custom_llm_api_instruct_base": "https://api2.heavy.ai/v1",
    }
    return change_iq_config(keys_to_change)
