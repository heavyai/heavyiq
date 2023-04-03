from flask import Blueprint, redirect, render_template, request, url_for
from flask.typing import ResponseReturnValue

routes_blueprint = Blueprint("index", __name__)


@routes_blueprint.route("/", methods=(["GET"]))
def index() -> ResponseReturnValue:
    return redirect("/api/v1/docs")
