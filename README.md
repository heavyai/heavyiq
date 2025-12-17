# HeavyIQ - Natural Language Interface for HeavyDB

HeavyIQ is a Python FastAPI application that provides a natural language interface to interact with [HeavyDB](https://github.com/heavyai/heavydb). It uses Large Language Models (LLMs) and vector databases for Retrieval Augmented Generation (RAG) to convert natural language questions into SQL queries and provide intelligent answers.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Bare Metal Installation](#bare-metal-installation)
- [Configuration](#configuration)
- [Running HeavyIQ](#running-heavyiq)
  - [Docker](#docker)
  - [Bare Metal](#bare-metal)
- [API Endpoints](#api-endpoints)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Architecture](#architecture)
- [Deployment](#deployment)

---

## Features

- **Natural Language to SQL**: Convert questions like "How many states have population over 1 million?" into valid SQL
- **RAG (Retrieval Augmented Generation)**: Enhance LLM responses with context from your database schema
- **Multiple Vector Store Backends**: Support for FAISS (in-memory, fast) and ChromaDB (persistent)
- **Streaming Responses**: Real-time streaming of LLM responses
- **Multi-process Safe**: File-based locking for concurrent FAISS access across Gunicorn workers
- **Table-level Security**: Respects HeavyDB access permissions
- **REST API & CLI**: Multiple interfaces for integration

---

## Prerequisites

### Required

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | 3.11 recommended |
| HeavyDB | 7.0+ | Running and accessible |
| LLM API | - | Heavy.ai LLM API, OpenAI, or Azure OpenAI |

### Optional

| Component | Version | Notes |
|-----------|---------|-------|
| Docker | 20.10+ | For containerized deployment |
| Docker Compose | 2.0+ | For multi-container orchestration |
| ChromaDB Server | 0.4+ | If using ChromaDB backend |

---

## Installation

### Docker Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/heavyai/heavyiq.git
   cd heavyiq
   ```

2. **Create configuration file**
   ```bash
   cp config.example.toml config.toml
   # Edit config.toml with your settings
   ```

3. **Build and start containers**
   ```bash
   # Build the Docker image
   docker compose -f docker/docker-compose.ubuntu.yml build

   # Start HeavyIQ
   docker compose -f docker/docker-compose.ubuntu.yml up -d
   ```

4. **Verify it's running**
   ```bash
   curl http://localhost:6279/docs
   ```

### Bare Metal Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/heavyai/heavyiq.git
   cd heavyiq
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   # For development (includes testing tools)
   pip install -r requirements-dev.txt
   ```

4. **Create configuration file**
   ```bash
   cp config.example.toml config.toml
   ```

5. **Configure your settings** (see [Configuration](#configuration))

6. **Run the application**
   ```bash
   # Development mode
   uvicorn 'heavyiq.api:create_app()' --host 0.0.0.0 --port 6279 --reload

   # Production mode
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker 'heavyiq.api:create_app()' --preload -b :6279
   ```

---

## Configuration

Create `config.toml` from the example:

```toml
[iq]
# HeavyDB Connection
heavydb_host = "localhost"
heavydb_port = 6274
heavydb_dbname = "heavyai"
heavydb_username = "admin"
heavydb_password = "HyperInteractive"

# LLM Configuration (choose one)
# Option 1: Heavy.ai LLM API
custom_llm_type = "API_VLLM"
custom_llm_api_base = "https://api.heavy.ai/v1"
heavylm_api_key = "your-api-key"

# Option 2: OpenAI
# openai_api_key = "sk-..."
# openai_gpt_model = "gpt-4"

# Option 3: Azure OpenAI
# custom_llm_type = "AZURE"
# custom_llm_azure_openai_api_base = "https://your-resource.openai.azure.com"
# custom_llm_azure_deployment_name = "your-deployment"

# RAG Configuration
enable_rag = true
rag_vectordb_type = "faiss"  # or "chroma"
rag_embed_server_base = "https://embeddings.heavy.ai/v1"

# Storage
data = "./storage"
```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `heavydb_host` | HeavyDB server hostname | `localhost` |
| `heavydb_port` | HeavyDB server port | `6274` |
| `custom_llm_type` | LLM type: `API`, `API_VLLM`, `AZURE` | - |
| `rag_vectordb_type` | Vector store: `faiss` or `chroma` | `faiss` |
| `enable_rag` | Enable RAG features | `true` |
| `port` | HeavyIQ API port | `6279` |

---

## Running HeavyIQ

### Docker

```bash
# Start
docker compose -f docker/docker-compose.ubuntu.yml up -d

# View logs
docker compose -f docker/docker-compose.ubuntu.yml logs -f

# Stop
docker compose -f docker/docker-compose.ubuntu.yml down

# Restart
docker compose -f docker/docker-compose.ubuntu.yml restart
```

### Bare Metal

```bash
# Development (with hot reload)
uvicorn 'heavyiq.api:create_app()' --host 0.0.0.0 --port 6279 --reload

# Production (with Gunicorn)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker 'heavyiq.api:create_app()' --preload -b :6279
```

---

## API Endpoints

Access the interactive API documentation at `http://localhost:6279/docs`

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Convert question to SQL and execute |
| POST | `/tables` | Get relevant tables for a question |
| POST | `/answer` | Get natural language answer |
| POST | `/auto/query` | Auto-detect tables and generate SQL |

### Streaming Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/stream/query` | Stream SQL generation |
| POST | `/stream/cot_query` | Stream with chain-of-thought |

### RAG Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/rag/documents/upload` | Upload documents for RAG |
| POST | `/rag/documents/ask` | Ask questions about documents |
| POST | `/rag/snippets/insert` | Add knowledge snippets |
| POST | `/rag/snippets/search` | Search snippets |
| POST | `/rag/snippets/list` | List all snippets |
| DELETE | `/rag/snippets/delete` | Delete a snippet |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/update-index` | Update vector store index |
| POST | `/llm/call` | Direct LLM invocation |
| GET | `/debug/db/session` | Get HeavyDB session (debug only) |

### Example Request

```bash
# Get a session ID
SESSION_ID=$(curl -s "http://localhost:6279/debug/db/session" | jq -r '.session_id')

# Ask a question
curl -X POST "http://localhost:6279/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many states have population over 1 million?",
    "session_id": "'$SESSION_ID'",
    "tables": ["heavyai_us_states"]
  }'
```

---

## CLI Usage

```bash
# Show help
python cli.py --help

# Database commands
python cli.py db session              # Get a HeavyDB session
python cli.py db tables               # List tables
python cli.py db table-info <table>   # Get table info

# RAG commands
python cli.py rag sync                # Sync table metadata to vector store
python cli.py rag search <query>      # Search vector store
```

---

## Testing

### Run Tests in Docker (Recommended)

```bash
# Start the container first
docker compose -f docker/docker-compose.ubuntu.yml up -d

# Run all tests (unit + e2e, skips integration)
./scripts/run-tests.sh

# Run only e2e API tests
./scripts/run-tests.sh --e2e

# Run only unit tests
./scripts/run-tests.sh --unit

# Run all tests including integration
./scripts/run-tests.sh --all

# Run specific test file
./scripts/run-tests.sh tests/heavyrag/test_faiss_vectorstore.py

# Run specific test with pattern
./scripts/run-tests.sh tests/ -k "test_concurrent_add" -v

# Run with coverage
./scripts/run-tests.sh --coverage
```

### Run Tests Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
./scripts/run-tests.sh --local

# Or use pytest directly
pytest tests/ --config-path config.toml -v
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `e2e` | End-to-end API tests (require full app stack) |
| `integration` | Integration tests (require external services) |
| (none) | Unit tests |

See [tests/README.md](tests/README.md) for comprehensive testing documentation.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### High-Level Flow

```
User Question
     │
     ▼
┌─────────────────┐
│   HeavyIQ API   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│ LCEL  │ │    RAG    │
│Chains │ │ (FAISS/   │
│       │ │ ChromaDB) │
└───┬───┘ └─────┬─────┘
    │           │
    └─────┬─────┘
          ▼
    ┌───────────┐
    │    LLM    │
    │ (Heavy.ai │
    │  /OpenAI) │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │  HeavyDB  │
    └───────────┘
```

### Key Components

- **LCEL Chains**: LangChain Expression Language for composable LLM workflows
- **RAG**: Retrieval Augmented Generation with FAISS or ChromaDB
- **File Locking**: Cross-process safe FAISS writes using `fcntl.flock()`
- **Gunicorn**: Production WSGI server with multiple workers

---

## Deployment

### Production with Gunicorn

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  'heavyiq.api:create_app()' \
  --preload \
  -b :6279 \
  --timeout 120
```

### Environment Variables

For FAISS thread safety (automatically set by HeavyIQ):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
```

### Docker Production

```bash
docker compose -f docker/docker-compose.ubuntu.yml up -d
```

---

## License

See [LICENSE](LICENSE) for details.
