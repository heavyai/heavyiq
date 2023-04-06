import connexion
from flask import Flask, redirect
from flask_cors import CORS

from modules.api import api_blueprint
from modules.routes import routes_blueprint


def redirect_ui() -> Flask.redirect:
    return redirect("/api/v1/ui/", code=302)


def get_app() -> Flask:
    app = connexion.FlaskApp(__name__)
    CORS(app.app)
    app.add_api("heavyanalyst.yaml", strict_validation=True)

    app.add_url_rule("/", "redirect_ui", redirect_ui)

    return app.app
