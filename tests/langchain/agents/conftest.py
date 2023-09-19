from unittest.mock import patch

import pytest
from confz import DataSource

from heavyiq.config.config_schema import HeavyIQConfig
from heavyiq.langchain import HeavyDB


@pytest.fixture(scope="package")
def mock_heavy_db():
    new_source = DataSource(
        data={
            "heavydb_dbname": "heavynl",
            "heavydb_username": "admin",
            "heavydb_password": "HyperInteractive",
            "heavydb_host": "10.2.1.33",
            "heavydb_port": "6274",
            "heavydb_protocol": "binary",
            "openai_api_key": "",
        }
    )
    try:
        # Use patch to mock the get_config function
        with HeavyIQConfig.change_config_sources(new_source):
            with patch("heavyiq.langchain.heavydb.get_config", return_value=HeavyIQConfig()):
                yield HeavyDB.from_env()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None

    print("Tearing heavydb mock client")
