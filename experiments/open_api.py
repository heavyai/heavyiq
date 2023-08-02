import os

from langchain.tools import OpenAPISpec, APIOperation
from langchain.chains import OpenAPIEndpointChain
from langchain.requests import Requests
from langchain.llms import OpenAI

from heavyiq.config import get_config

os.environ["OPENAI_API_KEY"] = get_config().openai_api_key


def main():
    spec = OpenAPISpec.from_url("http://localhost:5000/api/v1/openapi.json")
    operation = APIOperation.from_openapi_spec(spec, "/query", "post")
    chain = OpenAPIEndpointChain.from_api_operation(operation, OpenAI(), requests=Requests(), verbose=True)
    chain.run("How many states in the usa_states table start with the letter A?")


if __name__ == "__main__":
    main()
