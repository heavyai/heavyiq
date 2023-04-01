from flask import Flask
from flask_cors import CORS

from modules.api import api_blueprint
from modules.routes import routes_blueprint


app = Flask(__name__)
CORS(app)
app.register_blueprint(routes_blueprint)
app.register_blueprint(api_blueprint)
