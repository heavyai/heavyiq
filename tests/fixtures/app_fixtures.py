from collections.abc import Generator
from unittest.mock import patch

# import faiss on test module is must or otherwise we should endup in segmentation fault error upon running tcs
import pytest

from heavyiq.config import HeavyIQConfig
from heavyiq.langchain import HeavyDB


@pytest.fixture(scope="function")
def session_id(heavyiq_config: HeavyIQConfig) -> Generator[str, None, None]:
    try:
        # Use patch to mock the get_config function
        with patch("heavyiq.langchain.heavydb.get_config", return_value=heavyiq_config):
            yield HeavyDB.create_session_id()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None
