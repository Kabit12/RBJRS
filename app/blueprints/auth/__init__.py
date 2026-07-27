"""
Authentication Blueprint
=========================
Handles user registration, login, logout, and session management.
Will be fully implemented in Milestone 2.
"""

from flask import Blueprint

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

from app.blueprints.auth import routes  # noqa: E402, F401

# Exempt Google callback from CSRF (Google posts directly; token verification provides security)
from app.extensions import csrf
csrf.exempt(routes.google_callback)

