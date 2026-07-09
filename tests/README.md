# Run Unit Tests:

### API / “end-to-end” coverage

- **`tests/api/`** exercises the FastAPI app with **`TestClient`**: real HTTP routing, middleware, and dependency wiring, but **HeavyDB is overridden** via `tests/api/conftest.py` and `tests/api/dependencies.py` (`HeavyDB.from_env()` behind patched config). So these are **API integration tests**, not full production end-to-end runs against a live HeavyDB + real LLM stack.
- **`tests/api/test_lcel.py`** hits LCEL JSON routes (`/api/v1/lcel/query`, `question`, etc.) and expects successful responses when the mocked DB layer works.
- **`tests/api/test_feedback.py`** covers `/api/v1/submit-feedback` with LangSmith mocked.
- Broader **unit** coverage lives under `tests/config`, `tests/langchain`, `tests/heavyrag`, etc.

### Configuration

- **`config.pytest.toml`** (committed at the repo root) is the default for pytest when you omit `--config-path`. It uses `API_VLLM` with a dummy base URL and an empty `rag_embed_server_base`; the suite patches vLLM discovery and installs **LlamaIndex `MockEmbedding`**, so unit tests do not require a reachable custom LLM or embeddings server.
- For local overrides (secrets, real DB hosts, etc.), copy it to **`config.test.toml`** (gitignored) and run with `--config-path=./config.test.toml`.

Example `config.test.toml` fields when you need a real HeavyDB or different paths:

```toml
[iq]

heavydb_host = "0.0.0.0"
heavydb_port = 6274
heavydb_dbname = "heavyai" # heavyai, overture_wa
heavydb_username = "admin"
heavydb_password = "heavyisveryfast"

metadata_index_dir = "testdb"
langsmith_project = ""
langsmith_api_key = ""
log_to_stdout = true

access_log_level = "INFO"
heavyiq_log_level = "INFO"

data = "/test-data" # path to the directory where test data resides
enable_rag = true
custom_llm_api_base = "https://api.heavy.ai/v1" # llama3.1 8B-Instruct
rag_embed_server_base = "https://embeddings.heavy.ai"
rag_database_uri = "sqlite:///test_ragdb.sqlite"
rag_vectordb_type = "faiss" # Default is set to "faiss" for test cases to avoid the need to run a Chroma server when using "chroma".
```

```bash
# Default (uses config.pytest.toml when present)
pytest tests --disable-warnings -rs -n 0 --maxfail=1

# Explicit config file
pytest tests --disable-warnings -rs --config-path="./config.test.toml" -n 0 --maxfail=1
```

### Sanity run (no HeavyDB)

Tests that need a live HeavyDB are tagged with `@pytest.mark.heavydb` (set at module level for `tests/langchain/test_heavydb.py`, `tests/test_metadata_index.py`, `tests/api/test_lcel.py`, `tests/heavyrag/*`, `tests/rag/test_facts.py`, `tests/heavyrag/chains/test_retrieve_facts_chain.py`).

`tests/conftest.py` does a fast TCP probe of the configured `heavydb_host:heavydb_port` and **auto-skips** these tests when HeavyDB is unreachable. To force-skip them without probing the host, pass `--no-heavydb`:

```bash
# Pure sanity run: skip every @pytest.mark.heavydb test up front.
pytest tests --disable-warnings -rs -n 0 --maxfail=1 --no-heavydb

# Same intent, but using pytest's marker selector.
pytest tests --disable-warnings -rs -n 0 --maxfail=1 -m "not heavydb"
```

Either form leaves config / langchain LLM / RAG-snippet-vectorstore unit tests running and skips the ones that would otherwise call `HeavyDB.from_env()` / `HeavyDB.from_env_async()`.

HeavyRAG tests use an embedded ChromaDB `PersistentClient` fixture (see `tests/heavyrag/conftest.py`), so a separate `chroma run` server is not required.

Run the integration tests with:

```bash
pytest tests/heavyrag/test_controller.py -rs --config-path config.test.toml -n 0 --maxfail=1
```


## Run Tests Inside Container

1. Build test container.

```bash
docker-compose build heavyiq-test
```

2. Run tests.

```bash
docker-compose up --abort-on-container-exit heavyiq-test
```
