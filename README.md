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

