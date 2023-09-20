import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def log_file(tmp_path: Path):
    temp_dir = tmp_path / "log_files"
    temp_dir.mkdir()

    log_file_path = temp_dir / "test.log"
    yield log_file_path
