import pytest
from typing import Iterator
from contextlib import contextmanager


@contextmanager
def assert_not_raises(exception: type[BaseException]) -> Iterator[None]:
    try:
        yield
    except exception as e:
        pytest.fail(f"Expected no exception, but got: {e}")
