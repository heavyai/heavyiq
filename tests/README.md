# HeavyIQ Testing Guide

## Overview

HeavyIQ uses pytest for testing with three test categories:

| Marker | Description | Requires |
|--------|-------------|----------|
| (none) | Unit tests | Nothing external |
| `e2e` | End-to-end API tests | Full app stack, HeavyDB |
| `integration` | Integration tests | External services (ChromaDB, embedding server) |

---

## Quick Start

### Run Tests in Docker (Recommended)

```bash
# Run all tests (unit + e2e, skips integration)
./scripts/run-tests.sh

# Run only e2e API tests
./scripts/run-tests.sh --e2e

# Run only unit tests (no e2e)
./scripts/run-tests.sh --unit

# Run all tests including integration
./scripts/run-tests.sh --all

# Run with coverage
./scripts/run-tests.sh --coverage
```

### Run Tests Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests locally
./scripts/run-tests.sh --local

# Or use pytest directly
pytest tests/ --config-path config.toml -v
```

---

## Test Script Options

```bash
./scripts/run-tests.sh [OPTIONS] [TEST_PATH]

Options:
  --container     Run tests in Docker container (default)
  --local         Run tests on host machine
  --unit          Run unit tests only (skips both integration & e2e)
  --e2e           Run e2e API endpoint tests only
  --integration   Run integration tests only
  --all           Run all tests (unit + integration + e2e)
  --coverage      Generate coverage report
  -h, --help      Show help message

Examples:
  ./scripts/run-tests.sh                                    # Default: unit + e2e
  ./scripts/run-tests.sh --all                              # All tests
  ./scripts/run-tests.sh --e2e                              # E2E API tests only
  ./scripts/run-tests.sh tests/heavyrag/                    # Specific directory
  ./scripts/run-tests.sh tests/api/test_lcel.py            # Specific file
  ./scripts/run-tests.sh tests/ -k "test_query" -v         # Pattern match
  ./scripts/run-tests.sh tests/ -k "test_concurrent" -v    # FAISS concurrency tests
```

---

## Prerequisites

### For Docker Tests

1. **Docker running**
   ```bash
   # If you get permission errors, add yourself to docker group
   newgrp docker
   ```

2. **HeavyDB accessible** (for e2e tests)
   - Default: `host.docker.internal:16274`
   - Configure in `config.container.toml`

3. **Container built**
   ```bash
   docker compose -f docker/docker-compose.ubuntu.yml build
   ```

### For Local Tests

1. **Python 3.10+** with virtual environment
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. **HeavyDB running** (for e2e tests)
   - Configure in `config.toml`

3. **ChromaDB server** (for integration tests only)
   ```bash
   chroma run --host 127.0.0.1 --port 6271 --path /tmp/chromadb
   ```

---

## Configuration

### Test Configuration Files

| File | Use Case |
|------|----------|
| `config.container.toml` | Docker tests (mounts as `config.toml`) |
| `config.toml` | Local tests |
| `config.test.toml` | CI/alternative test config |

### Example `config.container.toml`

```toml
[iq]
data = "/code/storage"
heavydb_host = "host.docker.internal"
heavydb_port = 16274
heavydb_dbname = "heavyai"
heavydb_username = "admin"
heavydb_password = "HyperInteractive"
custom_llm_api_base = "https://api.heavy.ai/v1"
rag_embed_server_base = "https://embeddings.heavy.ai/v1"
custom_llm_type = "API_VLLM"
rag_vectordb_type = "faiss"
enable_rag = true
```

---

## Test Categories

### Unit Tests

Tests that don't require external services:

```bash
./scripts/run-tests.sh --unit
```

Location: `tests/` (files not in `tests/api/`)

Examples:
- `tests/heavyrag/test_faiss_vectorstore.py` - FAISS vector store
- `tests/lcel/chains/test_utils.py` - Chain utilities
- `tests/langchain/llms/test_llms.py` - LLM configuration

### E2E Tests

End-to-end API tests that require the full application stack:

```bash
./scripts/run-tests.sh --e2e
```

Location: `tests/api/`

Examples:
- `tests/api/test_lcel.py` - LCEL endpoint tests
- `tests/api/test_rag_router.py` - RAG endpoint tests

### Integration Tests

Tests requiring external services like ChromaDB:

```bash
./scripts/run-tests.sh --integration
```

Marker: `@pytest.mark.integration`

---

## Running Specific Tests

### By File Path

```bash
./scripts/run-tests.sh tests/heavyrag/test_faiss_vectorstore.py
```

### By Test Name Pattern

```bash
# Match test names containing "concurrent"
./scripts/run-tests.sh tests/ -k "test_concurrent" -v

# Match specific test
./scripts/run-tests.sh tests/ -k "test_concurrent_add_operations" -v
```

### By Test Class

```bash
./scripts/run-tests.sh "tests/heavyrag/test_faiss_vectorstore.py::TestFaissFileLock" -v
```

### By Exact Test

```bash
./scripts/run-tests.sh "tests/heavyrag/test_faiss_vectorstore.py::TestFaissFileLock::test_write_lock_blocks_readers" -v
```

---

## Common Test Scenarios

### FAISS Concurrency Tests

```bash
# Test file locking mechanism
./scripts/run-tests.sh tests/heavyrag/test_faiss_vectorstore.py -k "TestFaissFileLock" -v

# Test concurrent writes
./scripts/run-tests.sh tests/heavyrag/test_faiss_vectorstore.py -k "TestFaissConcurrentWrites" -v
```

### LCEL Chain Tests

```bash
./scripts/run-tests.sh tests/lcel/ -v
```

### LLM Configuration Tests

```bash
./scripts/run-tests.sh tests/langchain/llms/test_llms.py -v
```

### API Endpoint Tests

```bash
./scripts/run-tests.sh tests/api/test_lcel.py -v
```

---

## Skipping Tests

Some tests are skipped by default:

```python
@pytest.mark.skip(reason="Reason for skipping")
def test_something():
    pass

@pytest.mark.integration  # Skipped unless --integration flag
def test_chromadb():
    pass
```

To run skipped tests:

```bash
# Run with skip plugin disabled
./scripts/run-tests.sh tests/ -p no:skip -v
```

---

## Fixtures

### Common Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `heavyiq_config` | module | Loaded HeavyIQConfig |
| `client` | module | FastAPI TestClient |
| `session_id` | function | HeavyDB session ID |
| `ragdb` | module | RAG database instance |
| `faiss_rag_controller` | module | FAISS/Chroma controller |

### RAG Fixtures (`tests/fixtures/rag_fixtures.py`)

```python
@pytest.fixture
def require_chromadb_server(heavyiq_config):
    """Skip if ChromaDB server not running."""
    ...

@pytest.fixture
async def pre_clean_table(ragdb, faiss_rag_controller):
    """Clean RAG index before each test."""
    ...
```

---

## Troubleshooting

### "Container not running"

```bash
# Start the container first
docker compose -f docker/docker-compose.ubuntu.yml up -d

# Then run tests
./scripts/run-tests.sh
```

### "ChromaDB server not running"

For integration tests:

```bash
# Option 1: Use FAISS instead (edit config.container.toml)
rag_vectordb_type = "faiss"

# Option 2: Start ChromaDB server
chroma run --host 127.0.0.1 --port 6271 --path /tmp/chromadb
```

### "Could not connect to HeavyDB"

Check that HeavyDB is running and accessible:

```bash
# For Docker tests, check host.docker.internal resolves
docker exec heavyiq-ubuntu ping -c 1 host.docker.internal

# Verify HeavyDB port
nc -zv host.docker.internal 16274
```

### Test Isolation Issues

If tests fail due to cached state:

```bash
# Clear pytest cache
rm -rf .pytest_cache __pycache__

# Clear FAISS global state (done automatically by fixtures)
```

---

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import patch, MagicMock

class TestMyFeature:
    @pytest.fixture
    def my_fixture(self):
        yield SomeResource()

    def test_something(self, my_fixture):
        result = my_fixture.do_something()
        assert result == expected
```

### E2E Test Example

```python
import pytest
from fastapi.testclient import TestClient
from heavyiq.config import HeavyIQConfig

pytestmark = pytest.mark.e2e

class TestMyEndpoint:
    @pytest.mark.anyio
    async def test_endpoint(self, client: TestClient, session_id: str):
        response = client.post("/my-endpoint", json={"session_id": session_id})
        assert response.status_code == 200
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
def test_chromadb_connection(require_chromadb_server):
    """This test requires ChromaDB server running."""
    # Test code here
```

---

## Coverage

Generate coverage report:

```bash
./scripts/run-tests.sh --coverage

# View HTML report
open htmlcov/index.html
```

---

## CI/CD

Tests are run in CI with:

```bash
./scripts/run-tests.sh --unit  # Fast, no external dependencies
```

For full test suite:

```bash
./scripts/run-tests.sh --all
```
