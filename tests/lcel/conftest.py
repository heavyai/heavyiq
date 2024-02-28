from unittest.mock import patch

import pytest

from heavyiq.config import HeavyIQConfig
from heavyiq.langchain import HeavyDB


@pytest.fixture(scope="module")
def session_id(heavyiq_config: HeavyIQConfig):
    try:
        # Use patch to mock the get_config function
        with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
            yield HeavyDB.create_session_id()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None
