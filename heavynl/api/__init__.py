from typing import Any

import connexion
from flask import Flask, Response, redirect, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO

from flask_session import Session
from heavynl.api.models import db
from heavynl.config import get_config
from heavynl.logging_utils import _app_logger, heavynl_logger

from .handlers.ws_handlers import ChatNamespace

socketio = SocketIO(manage_session=False, cors_allowed_origins="*")


def set_config(app: Flask):
    """
    Here module related configs are being appended to app config.
    """
    app.secret_key = "123456789012345678901234"  # TODO: fixme
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sessions.db"


def get_app(config_path: str = "./config.toml") -> Flask:
    """
    Gets the Flask App instance.

    Args:
        config_path (str, optional): Config path. Defaults to "./config.toml".

    Returns:
        Flask: Final Flask instance after in-corporated all the modules.
    """
    get_config(config_path)  # loads config using specified path

    # see heavynl/ui/README.md
    app = connexion.FlaskApp(
        __name__,
        server_args={"static_folder": "../ui/build", "static_url_path": "/"},
    )

    flask_app: Flask = app.app
    set_config(flask_app)

    CORS(flask_app, resources={r"*": {"origins": "*"}}, origins="http://localhost:3000")  # TODO: allow specific origins

    sess = Session()
    sess.init_app(flask_app)
    db.init_app(flask_app)

    @flask_app.route("/index")
    def index() -> Any:
        return flask_app.send_static_file("index.html")

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyanalyst.yaml", strict_validation=True, base_path="/api/v1")

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    ui_path = "/api/v1/ui/"
    open_ai_path = "/api/v1/openapi.json"

    @flask_app.after_request  # type: ignore
    def log_response(response: Response) -> Response:
        """
        Called after the handler method of the corresponding endpoint.

        This method specifically logs the http request calls using app_logger.
        """
        request.response = response  # type: ignore
        _app_logger.info("", extra={"response": response})
        # do nothing for ui request
        if ui_path in request.path or open_ai_path in request.path:
            return response
        # otherwise log response of every request.
        try:
            heavynl_logger.debug("Response Content: %s", response.get_data(as_text=True))
        except RuntimeError:
            # skip writing response data in-case of runtime error
            pass
        heavynl_logger.debug("Response Status Code: %d", response.status_code)
        return response

    @flask_app.before_request  # type: ignore
    def log_request():
        """
        Called before the handler method of the corresponding endpoint.
        Here we just log the request body using heavynl_logger.
        """
        # do nothing for ui request
        if ui_path in request.path or open_ai_path in request.path:
            return
        heavynl_logger.debug("Request Path: %s", request.path)
        heavynl_logger.debug("Request Body: %s", request.get_data(as_text=True))

    # socket endpoints
    socketio.init_app(flask_app)

    socketio.on_namespace(ChatNamespace(namespace="/chat"))

    with flask_app.app_context():
        db.create_all()

    return flask_app  # type: ignore
