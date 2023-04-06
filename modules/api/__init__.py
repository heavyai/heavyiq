import connexion
from flask import Flask, redirect, Response
from flask_cors import CORS


def get_app() -> Flask:
    app = connexion.FlaskApp(__name__)
    CORS(app.app)
    app.add_api("heavyanalyst.yaml", strict_validation=True)

    app.add_url_rule("/", "redirect_ui", lambda: redirect("/api/v1/ui/", code=302))

    return app.app
