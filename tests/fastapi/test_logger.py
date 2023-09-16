import pytest
from tests import assert_not_raises
from heavyiq.logging_utils import iq_logger, init_logs


def test_should_raise_exception_for_iq_logger_if_accessed_before_calling_init_logs_function():
    with pytest.raises(AttributeError) as exc_info:
        iq_logger.info("Hi")

    assert str(exc_info.value) == "Logging has not yet been initialized."


def test_should_pass_iq_logger_if_accessed_after_calling_init_logs_function():
    init_logs()
    with assert_not_raises(AttributeError):
        iq_logger.info("This is a test message.")
