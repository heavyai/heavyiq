# HeavyIQ Architecture

## Introduction

HeavyIQ is a Python application that provides a natural language interface to [HeavyDB](https://github.com/heavyai/heavydb) using Large Language Models (LLMs) and vector databases for Retrieval Augmented Generation (RAG). It converts natural language questions into SQL queries and provides intelligent answers.

**HeavyIQ is available via three interfaces:**
- **REST API**: OpenAPI 3.0 compliant endpoints via FastAPI
- **Command-Line Interface**: Interactive CLI via Click
- **Python Module**: Direct programmatic access

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Request                               │
│         "How many states have population over 1 million?"               │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                             │
│                        (heavyiq/api/__init__.py)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   /query    │  │  /tables    │  │  /answer    │  │ /rag/...    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LCEL Chains Layer                               │
│                    (heavyiq/lcel/chains/heavydb/)                       │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  query_chain    │  │  table_chain    │  │  answer_chain   │         │
│  │ (NL → SQL)      │  │ (NL → Tables)   │  │ (SQL → Answer)  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                │                                        │
│                                ▼                                        │
│                    ┌─────────────────────┐                              │
│                    │   LLM Runnable      │                              │
│                    │ (ConfigurableField) │                              │
│                    └─────────┬───────────┘                              │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Heavy.ai LLM  │  │     OpenAI      │  │   Azure OpenAI  │
│   (API_VLLM)    │  │                 │  │                 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     HeavyDB     │
                    │ (Execute SQL)   │
                    └─────────────────┘
```

---

## Key Components

### 1. FastAPI Application (`heavyiq/api/`)

The REST API layer built with FastAPI:

| Component | Path | Description |
|-----------|------|-------------|
| `__init__.py` | `heavyiq/api/` | App factory, middleware, startup hooks |
| `routes/` | `heavyiq/api/routes/` | Endpoint definitions |
| `handlers/` | `heavyiq/api/handlers/` | Business logic for endpoints |
| `models/` | `heavyiq/api/models/` | Pydantic request/response models |
| `decorators.py` | `heavyiq/api/handlers/` | `@with_db`, `@with_feedback_id` |

### 2. LCEL Chains (`heavyiq/lcel/`)

LangChain Expression Language (LCEL) for composable LLM workflows:

```
heavyiq/lcel/
├── chains/
│   └── heavydb/
│       ├── query_chain.py    # Natural language → SQL
│       ├── table_chain.py    # Natural language → Relevant tables
│       └── answer_chain.py   # SQL results → Natural language answer
├── llms/
│   └── __init__.py           # LLM runnable with ConfigurableField
├── prompts/
│   └── heavydb/              # Prompt templates
└── output_parsers/           # Parse LLM outputs
```

**LLM Configuration with ConfigurableField:**

```python
# The LLM is configurable at runtime
llm_runnable = nl_to_sql_llm.configurable_alternatives(
    ConfigurableField(id="llm_type"),
    default_key="nl_to_sql",
    nl_to_tables=nl_to_tables_llm,
    sql_to_answer=sql_to_answer_llm,
)
```

### 3. RAG System (`heavyrag/`)

Retrieval Augmented Generation for enhanced context:

```
heavyrag/
├── vector_stores/
│   ├── faiss.py              # FAISS with file locking
│   └── chroma.py             # ChromaDB client
├── controller.py             # RAG controller (FAISS/Chroma)
├── database/                 # SQLite for metadata
├── embed.py                  # Embedding model
└── ingest.py                 # Document ingestion
```

### 4. HeavyDB Integration (`heavyiq/langchain/`)

Database abstraction and schema introspection:

```
heavyiq/langchain/
├── heavydb.py                # HeavyDB class wrapper
├── heavydb_utils.py          # SQL validation, execution
└── index/
    └── heavydb/
        └── heavydb_metadata_index.py  # Table metadata indexing
```

---

## FAISS File Locking (Multi-Process Safety)

HeavyIQ uses file-based locking for concurrent FAISS access across Gunicorn workers:

```
┌──────────────────┐
│  Gunicorn Worker │──┐
└──────────────────┘  │
┌──────────────────┐  │    ┌─────────────────┐
│  Gunicorn Worker │──┼───▶│  .faiss.lock    │──▶ faiss_index.bin
└──────────────────┘  │    │  (fcntl.flock)  │
┌──────────────────┐  │    └─────────────────┘
│  Gunicorn Worker │──┘
└──────────────────┘
```

**Lock Types:**

| Operation | Lock Type | Concurrent Access |
|-----------|-----------|-------------------|
| `query()` | Read (shared) | ✅ Multiple readers |
| `reload()` | Read (shared) | ✅ Multiple readers |
| `add()` | Write (exclusive) | ❌ One at a time |
| `remove()` | Write (exclusive) | ❌ One at a time |
| `persist()` | Write (exclusive) | ❌ One at a time |

**Implementation:**

```python
class FaissFileLock:
    @contextmanager
    def read_lock(self):
        fcntl.flock(fd, fcntl.LOCK_SH)  # Shared lock
        yield
        fcntl.flock(fd, fcntl.LOCK_UN)

    @contextmanager
    def write_lock(self):
        fcntl.flock(fd, fcntl.LOCK_EX)  # Exclusive lock
        yield
        fcntl.flock(fd, fcntl.LOCK_UN)
```

---

## Vector Store Backends

HeavyIQ supports two vector store backends:

| Feature | FAISS | ChromaDB |
|---------|-------|----------|
| Storage | In-memory + file | Persistent |
| Speed | ⚡ Very fast | Slower |
| Memory | High (loads to RAM) | Lower |
| Multi-process | File locking | HTTP server |
| Setup | No server needed | Requires server |
| Use Case | Smaller datasets | Larger datasets |

**Configuration:**

```toml
# config.toml
rag_vectordb_type = "faiss"  # or "chroma"
rag_faiss_persist_dir = "./storage/rag_storage/faiss"
rag_chromadb_server_base = "http://localhost:6271"
```

---

## Request Flow

### Example: `/query` Endpoint

```
1. User Request
   POST /query {"question": "...", "tables": [...], "session_id": "..."}
   
2. Validation & Auth
   └─▶ with_db decorator validates session, tables
   
3. LCEL Chain Execution
   └─▶ query_chain.ainvoke({
         "question": question,
         "table_info": table_schema,
         "top_k": 5,
       })
   
4. LLM Call
   └─▶ OverrideVLLMOpenAI.ainvoke(prompt)
   
5. SQL Execution
   └─▶ heavydb.run(sql_query)
   
6. Response
   └─▶ {"sql": "SELECT ...", "output": [...]}
```

---

## Configuration

### Configuration Loading Order

1. Default values in `HeavyIQConfig`
2. `config.toml` file (via `--config-path` or auto-detect)
3. Environment variables (prefixed with `HEAVYIQ_`)

### Key Configuration Options

```toml
[iq]
# HeavyDB Connection
heavydb_host = "localhost"
heavydb_port = 6274
heavydb_dbname = "heavyai"

# LLM Configuration
custom_llm_type = "API_VLLM"      # API, API_VLLM, or AZURE
custom_llm_api_base = "https://api.heavy.ai/v1"
custom_llm_api_context_window = 8192

# RAG Configuration
enable_rag = true
rag_vectordb_type = "faiss"       # faiss or chroma
rag_embed_server_base = "https://embeddings.heavy.ai/v1"
rag_embed_dimension = 1024

# Storage
data = "./storage"
```

---

## Security

### Table-Level Access Control

1. **Session Authentication**: HeavyDB validates user permissions
2. **Allowed Tables Filter**: API enforces `allowed_tables` parameter
3. **Metadata Prefiltering**: RAG queries only include permitted tables

```python
# In decorators.py
@with_db
async def handler(request, db):
    # Validates request.allowed_tables against db.get_usable_table_names()
    # Raises 400 if unknown tables are specified
```

---

## Deployment

### Production with Gunicorn

```bash
# Set FAISS threading (automatically set by HeavyIQ on startup)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run with 4 workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  'heavyiq.api:create_app()' \
  --preload -b :6279 --timeout 120
```

### Docker

**Start HeavyIQ server (background):**
```bash
./scripts/start-heavyiq.sh -d
```

**Start with rebuild:**
```bash
./scripts/start-heavyiq.sh -d --build
```

**Stop HeavyIQ server:**
```bash
./scripts/stop-heavyiq.sh
```

**Stop and remove volumes:**
```bash
./scripts/stop-heavyiq.sh --clean
```

**View logs:**
```bash
docker logs -f heavyiq-ubuntu
```

**Execute commands inside container:**
```bash
docker exec -it heavyiq-ubuntu bash
```

---

## Directory Structure

```
heavyiq/
├── heavyiq/                  # Main application
│   ├── api/                  # FastAPI application
│   ├── lcel/                 # LCEL chains, prompts, parsers
│   ├── langchain/            # HeavyDB integration, legacy chains
│   ├── config/               # Configuration management
│   └── rag/                  # RAG router
├── heavyrag/                 # RAG module
│   ├── vector_stores/        # FAISS, ChromaDB implementations
│   ├── database/             # SQLite metadata storage
│   └── controller.py         # RAG controller
├── docker/                   # Docker configuration
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
└── config.toml               # Configuration file
```

---

## Logging

HeavyIQ uses Loguru for structured logging:

```python
from heavyiq.loguru_logging import logger

logger.info("Processing query", question=question)
logger.debug("SQL generated", sql=sql)
```

Logs are stored in `storage/log/` with rotation.

---

## Testing

See [tests/README.md](tests/README.md) for testing documentation.

```bash
# Run all tests
./scripts/run-tests.sh --all
# Run endpoint tests
./scripts/run-tests.sh --e2e

# Run specific test
./scripts/run-tests.sh tests/heavyrag/test_faiss_vectorstore.py -k "test_concurrent" -v
```
