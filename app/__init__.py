# app/__init__.py
from flask import Flask
from .routes import spc_routes
import os

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.register_blueprint(spc_routes)
    # Ensure output folder exists (using static_folder path)
    with app.app_context():
        outputs_dir = os.path.join(app.static_folder or "static", "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
    return app
