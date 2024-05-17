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


def validate_config(iq: HeavyIQConfig) -> AppConfig:
    """
    Validates app config after it's creation.
    This involves basic validation of configs and creation of directory mentioned if not exists.
    """

    if iq == HeavyIQConfig():
        print("No config provided. Skipping validation until license is loaded and config updated per edition used.")
    else:
        if iq.custom_llm_type is None or iq.custom_llm_type == "AZURE":
            if iq.custom_llm_type == "AZURE" and (
                iq.custom_llm_azure_deployment_name.strip() == ""
                or iq.custom_llm_azure_openai_api_base.strip() == ""
                or iq.custom_llm_azure_openai_api_version.strip() == ""
            ):
                raise ValueError(
                    "Custom LLM type is set to 'AZURE', but deployment name or API base URL or API Version is not set."
                )
            openai_client = OpenAI(api_key=iq.openai_api_key)
            if iq.custom_llm_type is None:
                try:
                    openai_client.models.list()
                except Exception as e:
                    raise ValueError(f"Unable to communicate with OpenAI API: {e}")
        elif iq.custom_llm_type == "API":
            if iq.custom_llm_api_base.strip() == "":
                raise ValueError("Custom LLM type is set to 'API', but API base URL is not set.")
        elif iq.custom_llm_type == "API_VLLM":
            if iq.custom_llm_api_base.strip() == "":
                raise ValueError("Custom LLM type is set to 'API_VLLM', but API base URL is not set.")
        else:
            raise ValueError(
                f"Invalid custom LLM type (valid options are AZURE or API or API_VLLM): {iq.custom_llm_type}"
            )
    

    return iq


def get_config(file: str = "./config.toml") -> HeavyIQConfig:
    """If called with a non-default file path, must be called before importing any other modules that use the config."""
    global _config

    with _config_lock:
        if _config:
            return _config
        app_config: AppConfig = AppConfig(config_sources=FileSource(file=file))  # type: ignore
        validate_config(app_config.iq)

        # Data setup things, move out of validate method
        if app_config.iq.data is None:
            if app_config.data:
                app_config.iq.data = app_config.data
            else:
                app_config.iq.data = "./storage"
        if not os.path.exists(app_config.iq.data):
            os.makedirs(app_config.iq.data)

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
        "custom_llm_api_base": "https://community.heavylm/heavy.ai/v1", # base url of API",
        "custom_llm_api_nl_to_sql_base": "https://community.heavylm/heavy.ai/v1",
        "custom_llm_api_sql_to_answer_base": "https://community.heavylm/heavy.ai/v1",
        "custom_llm_api_nl_to_tables_base": "https://community.heavylm/heavy.ai/v1",
        "custom_llm_api_tables_to_questions_base": "https://community.heavylm/heavy.ai/v1",
        "custom_llm_api_instruct_base": "https://community.heavylm/heavy.ai/v1",
    }

    change_success = change_iq_config(keys_to_change)

    global _no_config_provided
    if _no_config_provided == True:
        print("No config provided, using defaults")
        # Additional things to override if no config was provided and it's the free edition
        no_config_defaults = {
            "heavydb_port": 6274,
            "custom_llm_type": "API_VLLM",
        }
        change_iq_config(no_config_defaults)

    global _config
    validate_config(_config)
    
    return change_success
