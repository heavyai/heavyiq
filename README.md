# HeavyIQ - Natural Language Interface for HeavyDB

HeavyIQ is a python module that serves as a natural language interface to interact with HeavyDB. The application uses [LangChain](https://python.langchain.com/en/latest/index.html) and Large Language Models to process user’s natural language questions and provide corresponding answers based on the data in the database. The module provides a simple REST API for querying the database using natural language questions.
The application also provides a CLI interface for quick interactions with the HeavyIQ agent.

## Build and Run with Docker Compose

Before buiding the HeavyIQ containers, please make sure to make necessay configurations on
`config.toml` and `config.test.toml` files.

1. Build the Docker image for HeavyIQ API and HeavyIQ test containers:

    ```bash
    docker-compose build
    ```

2. Run HeavyIQ API container:

    ```bash
    docker-compose up heavyiq-api
    ```

3. Run tests.
   
   ```bash
   docker-compose up --abort-on-container-exit heavyiq-test
   ```

## Setup

1. If you don’t have Python >=3.10 installed, [install it from here](https://www.python.org/downloads/).

2. Clone this repository.

3. Navigate into the project directory:

```bash
$ cd heavyiq
```

4. Create a new virtual environment:

```bash
$ python3.10 -m venv venv
$ . venv/bin/activate
```

5. Install the requirements:

```bash
$ pip install -r requirements.txt -r requirements-dev.txt
# it may be necessary to `export HNSWLIB_NO_NATIVE=1` to install chromadb on Mac
```

6. Make a copy of the example config file:

```bash
$ cp config.example.toml config.toml
```

7. Add your [API key](https://beta.openai.com/account/api-keys) and HeavyDB Credentials to the newly created `config.toml` file.

8. Run the app:

```bash
$ uvicorn app:app --reload
```

You should now be able to access the API Documentation at [http://localhost:5000](http://localhost:5000)!

## Using the CLI
```bash
$ python cli.py --help
```

## Integration Tests
```bash
# App must be running locally
$ python -m unittest discover
```

or

Use `pytest`

```bash
# runs testcases specific to fastapi
$ pytest tests/fastapi --disable-warnings
```

Always run the testcases in sequential order by explicitly specifying the `-n=0` option, or otherwise we might endup with errors.

```bash
pytest tests/api/test_lcel.py  -rs --config-path config.test.toml -n 0 --maxfail=1 --disable-warnings
```

## Deployment

### Create Obfuscated Build

```bash
pyarmor gen ./modules
cp app.py ./dist/app.py
```

### Run Production Server Process

```bash
$ gunicorn -w 4 -k uvicorn.workers.UvicornWorker 'heavyiq.api:create_app("./path/to/heavy.conf")' --preload
```

Port can be specified in above command with command line flag `-b :8080`

## Gunicorn Hook Lifecycle

When running with `gunicorn --preload`, the following hooks are executed in order:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GUNICORN HOOK EXECUTION ORDER                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MASTER PROCESS STARTUP                                                  │
│  ════════════════════                                                    │
│                                                                          │
│  1. on_starting(server)          ◄── Called FIRST, before anything      │
│     │                                                                    │
│     ▼                                                                    │
│  2. [If --preload] Load app      ◄── create_app() runs here             │
│     │   └── app_initialize()                                            │
│     │                                                                    │
│     ▼                                                                    │
│  3. on_reload(server)            ◄── Only on reload, not initial start  │
│     │                                                                    │
│     ▼                                                                    │
│  FOR EACH WORKER:                                                        │
│  ════════════════                                                        │
│                                                                          │
│  4. pre_fork(server, worker)     ◄── Just BEFORE fork()                 │
│     │                                                                    │
│     ▼                                                                    │
│  5. ═══════ FORK() ═══════       ◄── Worker process created             │
│     │                                                                    │
│     ▼                                                                    │
│  6. post_fork(server, worker)    ◄── Just AFTER fork(), in WORKER       │
│     │                                                                    │
│     ▼                                                                    │
│  7. post_worker_init(worker)     ◄── After worker fully initialized     │
│     │                                                                    │
│     ▼                                                                    │
│  8. [Worker starts serving]      ◄── FastAPI lifespan startup runs      │
│                                                                          │
│  SHUTDOWN                                                                │
│  ════════                                                                │
│                                                                          │
│  9. worker_exit(server, worker)  ◄── When worker exits                  │
│     │                                                                    │
│     ▼                                                                    │
│  10. on_exit(server)             ◄── When master exits (cleanup)        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### HeavyIQ Initialization Flow with `--preload`

```
# 1. on_starting() - Master process
#    └── Starts ChromaDB server thread (if enabled)
#    └── Starts license check background thread

# 2. create_app() - Master process (--preload)
#    └── app_initialize()
#        └── start_manager()      # SharedState manager for cross-process state
#        └── rag_create_tables()  # SQLite tables for RAG

# 3. pre_fork() - Master process (for each worker)
#    └── gc.disable()             # Prevent GC issues during fork

# 4. FORK - Worker process created

# 5. post_fork() - Worker process
#    └── shared_state.connect_to_manager()  # Connect to shared state
#    └── rag_initialize()                   # Initialize FAISS/ChromaDB
#    └── gc.enable()                        # Re-enable garbage collection

# 6. FastAPI lifespan startup - Worker process
#    └── (RAG already initialized in post_fork, skipped)
#    └── Start telemetry background task
```

### RAG Initialization by Vector Store Type

| Step | FAISS | ChromaDB |
|------|-------|----------|
| **Master (pre-fork)** | `rag_create_tables()` | `rag_create_tables()` |
| **Worker (post-fork)** | `rag_initialize()` | `rag_initialize()` |
| **FastAPI startup** | Skip (already done) | Skip (already done) |

Both vector stores are initialized **after fork** for consistency. This ensures each worker has its own clean instance.

### Running with Uvicorn (Standalone)

When running with uvicorn directly (without gunicorn), the `post_fork` hook is not called. Instead, RAG initialization happens in the FastAPI lifespan startup event:

```bash
$ uvicorn 'heavyiq.api:create_app()' --host 0.0.0.0 --port 8000
```

```
# 1. create_app()
#    └── app_initialize()
#        └── rag_create_tables()

# 2. FastAPI lifespan startup
#    └── rag_initialize()  # Called here since post_fork didn't run
```

