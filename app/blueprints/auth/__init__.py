"""
Authentication Blueprint
=========================
Handles user registration, login, logout, and session management.
Will be fully implemented in Milestone 2.
"""

from flask import Blueprint

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

from app.blueprints.auth import routes  # noqa: E402, F401
