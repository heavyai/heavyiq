import json
import os
import sys

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


def get_spec(spec_path: str = "http://localhost:8000/openapi.json") -> ReducedOpenAPISpec:
    """
    return the spec dict from swagger url.
    """
    response = requests.get(spec_path, verify=False)
    spec_dict = json.loads(response.content)
    return reduce_openapi_spec(spec_dict)


def main():
    requests_wrapper = RequestsWrapper()
    model = get_config().openai_gpt_model
    llm = get_chat_llm(model=model)

    heavyiq_agent = planner.create_openapi_agent(get_spec(), requests_wrapper, llm)
    user_query = "How many states in the usa_states table start with the letter A?"
    heavyiq_agent.run(user_query)


if __name__ == "__main__":
    main()
