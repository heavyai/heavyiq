import os
from typing import Any

import connexion
from flask import Flask
from flask_cors import CORS

from heavynl.config import get_config


def get_app(config_path: str = "./config.toml") -> Flask:
    get_config(config_path)  # loads config using specified path

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
