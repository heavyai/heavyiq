# HeavyNL - Natural Language Interface for HeavyDB

HeavyNL is a python module that serves as a natural language interface to interact with HeavyDB. The application uses [LangChain](https://python.langchain.com/en/latest/index.html) and Large Language Models to process user’s natural language questions and provide corresponding answers based on the data in the database. The module provides a simple REST API for querying the database using natural language questions.
The application also provides a CLI interface for quick interactions with the HeavyNL agent.

## Setup

1. If you don’t have Python >=3.10 installed, [install it from here](https://www.python.org/downloads/).

2. Clone this repository.

3. Navigate into the project directory:

```bash
$ cd heavynl
```

4. Create a new virtual environment:

```bash
$ python3.10 -m venv venv
$ . venv/bin/activate
```

5. Install the requirements:

```bash
$ pip install -r requirements.txt
```

6. Make a copy of the example environment variables file:

```bash
$ cp .env.example .env
```

7. Add your [API key](https://beta.openai.com/account/api-keys) and HeavyDB Credentials to the newly created `.env` file.

8. Run the app:

```bash
$ flask run
```

You should now be able to access the app at [http://localhost:5000](http://localhost:5000)!
API Documents (Swagger) can be found at [http://localhost:5000/api/v1/docs/](http://localhost:5000/api/v1/docs/)

## Using the CLI
```bash
$ python cli.py "Your question here"
$ python cli.py --help
```

