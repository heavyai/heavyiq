import json
import os
import sys
import argparse
from typing import Any

import requests
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec, reduce_openapi_spec
from langchain.requests import RequestsWrapper

# append the current directory to the sys path, so that you could run this script
# directly from the root folder like,
# python experiments/open_api.py
sys.path.append(".")

from heavyiq.config import get_config
from heavyiq.langchain.llms import get_chat_llm

os.environ["OPENAI_API_KEY"] = get_config().openai_api_key


def parse_arguments() -> Any:
    """
    Parse Input arguments.
    """
    parser = argparse.ArgumentParser(description="OpenAPI Spec Planner and Executor")

    # Define a positional argument for the OpenAPI spec file path
    parser.add_argument(
        "spec_file",
        metavar="OPENAPI_SPEC_URL",
        type=str,
        help="OpenAPI spec file URL (e.g., http://localhost:8000/openapi.json)",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure that exactly one argument (the spec file path) is provided
    if not args.spec_file:
        parser.error("Please provide the path to the OpenAPI spec file.")
    return args


def get_spec(spec_file_path: str) -> ReducedOpenAPISpec:
    """
    return the spec dict from swagger url.
    """
    response = requests.get(spec_file_path, verify=False)
    spec_dict = json.loads(response.content)
    return reduce_openapi_spec(spec_dict)


def main(spec_file_path: str):
    requests_wrapper = RequestsWrapper()
    model = get_config().openai_gpt_model
    llm = get_chat_llm(model=model)

    heavyiq_agent = planner.create_openapi_agent(get_spec(spec_file_path), requests_wrapper, llm)
    user_query = "How many states in the usa_states table start with the letter A?"
    heavyiq_agent.run(user_query)


if __name__ == "__main__":
    args = parse_arguments()
    spec_file_path = args.spec_file
    main(spec_file_path)
