from typing import Any

import connexion
from flask import Flask, redirect
from flask_cors import CORS
from heavydb.thrift.ttypes import TDBException
from connexion.exceptions import OAuthProblem

from modules.langchain import HeavyDB


def apikey_auth(session_id: str, required_scopes: list[str]) -> dict[str, Any]:
    heavydb = HeavyDB.from_env()
    try:
        heavydb.validate_session(session_id)
    except TDBException as e:
        raise OAuthProblem(e.error_msg)
    return {"uid": session_id}


def get_app() -> Flask:
    app = connexion.FlaskApp(__name__)
    CORS(app.app)

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyanalyst.yaml", strict_validation=True, base_path="/api/v1")

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    return app.app
