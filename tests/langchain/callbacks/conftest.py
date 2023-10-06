import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def log_file(tmp_path: Path, request):
    """
    Fixture supposed to create and return log file path corresponding to the test name.
    """
    file_name = request.node.name + ".log"
    temp_dir = tmp_path / "log_files"
    temp_dir.mkdir()

    log_file_path = temp_dir / file_name
    yield log_file_path
