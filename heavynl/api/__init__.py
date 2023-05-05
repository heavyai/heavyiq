import os
from typing import Any

import connexion
from flask import Flask
from flask_cors import CORS

from heavynl.config import config
from heavynl.langchain.index import get_heavydb_index

os.environ["OPENAI_API_KEY"] = config.openai_api_key


def get_app() -> Flask:
    # eagerly load the index
    get_heavydb_index()

    # see heavynl/ui/README.md
    app = connexion.FlaskApp(__name__, server_args={"static_folder": "../ui/build", "static_url_path": "/"})
    CORS(app.app)

    @app.app.route("/")
    def index() -> Any:
        return app.app.send_static_file("index.html")

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyanalyst.yaml", strict_validation=True, base_path="/api/v1")

    return app.app
