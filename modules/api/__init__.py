import os

import connexion
from flask import Flask, redirect
from flask_cors import CORS

from modules.config import config
from modules.langchain.index import get_heavydb_index

os.environ["OPENAI_API_KEY"] = config.openai_api_key


def get_app() -> Flask:
    # eagerly load the index
    get_heavydb_index()

    app = connexion.FlaskApp(__name__)
    CORS(app.app)

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyanalyst.yaml", strict_validation=True, base_path="/api/v1")

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    return app.app
