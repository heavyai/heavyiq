import os

import connexion
from flask import Flask, redirect
from flask_cors import CORS

from heavynl.config import get_config


def get_app(config_path: str = "./config.toml") -> Flask:
    get_config(config_path)  # loads config using specified path

    app = connexion.FlaskApp(__name__)
    CORS(app.app)

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyanalyst.yaml", strict_validation=True, base_path="/api/v1")

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    return app.app
