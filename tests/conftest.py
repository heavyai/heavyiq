# Import faiss FIRST to avoid static TLS exhaustion error
# This must be the first import before any other heavy libraries (numpy, torch, etc.)
# See: https://github.com/facebookresearch/faiss/issues/2595
try:
    import faiss  # noqa: F401
except ImportError:
    pass  # FAISS not installed, skip

import asyncio
import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from heavyiq.config import get_config


def pytest_addoption(parser):
    parser.addoption("--config-path", action="store")


# Global variable to store the config file path
CONFIG_FILE = None


def pytest_configure(config):
    global CONFIG_FILE
    CONFIG_FILE = config.getoption("--config-path")


@pytest.fixture(scope="package")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def config_file_path(pytestconfig):
    """
    Returns test config file path which gets passed to `get_config` method.
    """
    config_path = pytestconfig.getoption("--config-path")
    if not config_path:
        pytest.skip(reason="--config-path option required!")
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
        yield get_config(config_file_path)


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
