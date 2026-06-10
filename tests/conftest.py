# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from heavyiq.config import get_config


def pytest_addoption(parser):
    parser.addoption(
        "--config-path",
        action="store",
        default=None,
        help="Path to HeavyIQ config TOML (defaults to config.pytest.toml at repo root when present).",
    )
    parser.addoption(
        "--no-heavydb",
        action="store_true",
        default=False,
        help="Skip tests marked @pytest.mark.heavydb without probing the HeavyDB host (sanity run).",
    )


# Global variable to store the config file path (used by tests.api.dependencies)
CONFIG_FILE = None


def _default_config_path() -> str | None:
    candidate = Path(__file__).resolve().parent.parent / "config.pytest.toml"
    return str(candidate) if candidate.is_file() else None


def pytest_configure(config):
    global CONFIG_FILE
    path = config.getoption("--config-path", default=None)
    if not path:
        path = _default_config_path()
    CONFIG_FILE = path


def _sync_heavyrag_embed_modules(mock_embed) -> None:
    """If heavyrag loaded before the mock was set, align module-level snapshots."""
    for mod_name in ("heavyrag.controller", "heavyrag.index"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "EMBED_MODEL"):
            setattr(mod, "EMBED_MODEL", mock_embed)
        if hasattr(mod, "embed_model"):
            setattr(mod, "embed_model", mock_embed)


def _install_offline_test_embedding(cfg) -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    import heavyrag.embed as embed_mod

    dim = int(getattr(cfg, "rag_embed_dimension", 1024) or 1024)
    mock = MockEmbedding(embed_dim=dim)
    embed_mod.EMBED_MODEL = mock
    embed_mod.EMBED_MODEL_MAX_LEN = 8192
    _sync_heavyrag_embed_modules(mock)


def _is_heavydb_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    """Quick TCP probe so we can skip DB-bound tests when no HeavyDB is reachable."""
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.heavydb tests when the configured HeavyDB host is unreachable."""
    skip_db = config.getoption("--no-heavydb", default=False)
    reason = "Skipping HeavyDB-bound tests (--no-heavydb)."

    if not skip_db:
        cfg_path = CONFIG_FILE
        if cfg_path and os.path.exists(cfg_path):
            try:
                with patch("heavyiq.config._config", new=None):
                    cfg = get_config(cfg_path)
                if not _is_heavydb_reachable(cfg.heavydb_host, cfg.heavydb_port):
                    skip_db = True
                    reason = f"HeavyDB unreachable at {cfg.heavydb_host}:{cfg.heavydb_port}; skipping @heavydb tests."
            except Exception as exc:  # config validation, etc.
                skip_db = True
                reason = f"Config not usable for HeavyDB probe ({exc}); skipping @heavydb tests."

    if skip_db:
        marker = pytest.mark.skip(reason=reason)
        for item in items:
            if "heavydb" in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session", autouse=True)
def _offline_vllm_discovery_for_tests():
    """Avoid HTTP to custom_llm / vLLM /models when resolving tokenizer or embed model names."""
    _name = "meta-llama/Meta-Llama-3-8B-Instruct"
    with (
        patch("heavyiq.langchain.llms.get_vllm_model_name", return_value=_name),
        patch("heavyiq.langchain.llms.get_vllm_max_model_len", return_value=8192),
        # utils binds get_vllm_model_name at import time; patch the local reference too.
        patch("heavyiq.langchain.utils.get_vllm_model_name", return_value=_name),
    ):
        yield


@pytest.fixture(scope="package")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def config_file_path(pytestconfig):
    """
    Returns test config file path which gets passed to `get_config` method.
    When --config-path is omitted, uses config.pytest.toml at the repo root if it exists.
    """
    config_path = pytestconfig.getoption("--config-path", default=None)
    if not config_path:
        config_path = _default_config_path()
    if not config_path:
        pytest.skip(reason="--config-path not set and config.pytest.toml not found at repo root.")
    if not os.path.exists(config_path):
        pytest.skip(reason="Config file not found!")
    return config_path


@pytest.fixture(scope="package", autouse=True)
def heavyiq_config(config_file_path):
    """
    Fixture that supposed to return HeavyIQConfig by reading the config from passed config_file_path fixture..
    """
    print(f"Getting config from {config_file_path}")
    with patch("heavyiq.config._config", new=None):
        cfg = get_config(config_file_path)
    _install_offline_test_embedding(cfg)
    yield cfg


@pytest.fixture(autouse=True, scope="session")
def print_total_time(request):
    start_time = time.time()

    yield  # This is where the test cases run

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time taken for all test cases: {total_time:.2f} seconds")


# declare this before any other fixtures with the "session" scope
# that may reference the event loop
@pytest.fixture(scope="session")
def event_loop():
    return asyncio.get_event_loop()
