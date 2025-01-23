# Run Unit Tests:

### Prerequesties

Set the relevant configuration flags required for establishing the HeavyDB connection. Some of the configuration flags that must be set before running the test cases include:

`config.test.toml`

```toml
[iq]

heavydb_host = "138.2.234.112" # 
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
pytest tests --disable-warnings -rs --config-path="./config.test.toml" -n 0 --maxfail=1
```

Before running the testcases for heavyrag, make sure to run the chromadb server beforehand like below,

```bash
chroma run --port 6271
```

And then run the integration tests,

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
