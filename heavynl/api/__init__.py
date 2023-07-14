from typing import Any

import connexion
from flask import Flask, redirect, request, Response
from flask_cors import CORS

from heavynl.config import get_config, get_heavydb_license_claims
from heavynl.logging_utils import _get_app_logger, get_heavynl_logger, init_logs


def get_app(config_path: str = "./config.toml") -> Flask:
    config = get_config(config_path)  # loads config using specified path
    init_logs()  # initializes logs using config

    app = connexion.FlaskApp(__name__, server_args={"static_folder": "../../public", "static_url_path": "/"})
    CORS(app.app)

    # for some reason the servers base url is not set to the full url, just "/api/v1"
    # maybe it's set when deployed for production? in any case, must be resolved for LLM schema discovery
    app.add_api("heavyiq-spec.yaml", strict_validation=True, base_path="/api/v1")

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    ui_path = "/api/v1/ui/"
    open_ai_path = "/api/v1/openapi.json"

    flask_app = app.app

    @flask_app.before_first_request
    def initialize():
        from heavynl.logging_utils import heavynl_logger as logger

        logger.info("Connecting to heavydb...")
        get_heavydb_license_claims(config)
        logger.info("Successfully connected to heavydb...")

    @flask_app.after_request  # type: ignore
    def log_response(response: Response) -> Response:
        """
        Called after the handler method of the corresponding endpoint.

        This method specifically logs the http request calls using app_logger.
        """
        request.response = response  # type: ignore
        _get_app_logger().info("", extra={"response": response, "request": request})
        # do nothing for ui request
        if ui_path in request.path or open_ai_path in request.path:
            return response
        # otherwise log response of every request.
        try:
            get_heavynl_logger().debug(
                "Response Content: %s", response.get_data(as_text=True), extra={"request": request}
            )
        except RuntimeError:
            # skip writing response data in-case of runtime error
            pass
        get_heavynl_logger().debug("Response Status Code: %d", response.status_code, extra={"request": request})
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
        get_heavynl_logger().debug("Request Path: %s", request.path)
        get_heavynl_logger().debug("Request Body: %s", request.get_data(as_text=True))

    return flask_app  # type: ignore
