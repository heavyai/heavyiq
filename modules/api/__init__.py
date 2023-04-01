from flask import Blueprint
from flask_restx import Api

api_blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
api = Api(
    api_blueprint,
    version="0.1",
    title="HeavyAnalyst API",
    description="A simple REST API for interacting with HeavyNL",
    doc="/docs/",
)

from . import routes  # noqa: E402
from . import rest_models  # noqa: E402
