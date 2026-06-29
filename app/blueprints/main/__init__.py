"""
Main Blueprint
===============
Handles the landing page and public-facing routes that don't
require authentication (home, about, etc.).
"""

from flask import Blueprint

main_bp = Blueprint('main', __name__)

from app.blueprints.main import routes  # noqa: E402, F401
