# HeavyIQ - Natural Language Interface for HeavyDB
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/heavyai/heavyiq/blob/master/LICENSE)
[![Security](https://img.shields.io/badge/Security-Report%20a%20Vulnerability-red.svg)](https://github.com/heavyai/heavyiq/blob/master/SECURITY.md)
[![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-blue?logo=github)](https://github.com/orgs/heavyai/discussions)



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

### Create Production Build

```bash
bash scripts/build_prod.sh --include_all_deps
```

`requirements.txt` is the authoritative dependency manifest. To explicitly
substitute a locally built pyheavydb wheel without changing that manifest:

```bash
bash scripts/build_prod.sh --include_all_deps \
  --pyheavydb-wheel=/absolute/path/to/pyheavydb-10.0.0-py3-none-any.whl
```

### Run Production Server Process

```bash
$ gunicorn -w 4 -k uvicorn.workers.UvicornWorker 'heavyiq.api:create_app("./path/to/heavy.conf")' --preload
```

Port can be specified in above command with command line flag `-b :8080`


## Security
> [!WARNING]
> **Do not report security vulnerabilities through public GitHub issues!**

NVIDIA takes security seriously. If you discover a vulnerability in useWhisper, **DO NOT open a public issue**. Use one of the private reporting channels described in [SECURITY.md](https://github.com/heavyai/heavyiq/blob/master/SECURITY.md).

## Support
Join the [HeavyAI GitHub Discussions](https://github.com/orgs/heavyai/discussions) to ask questions, share feedback, and report issues. HeavyAI maintainers review issues, discussions, and pull requests on a best effort basis without guaranteed response timelines.
  
## License
Apache 2.0. See [LICENSE](https://github.com/heavyai/heavyiq/blob/main/LICENSE).

