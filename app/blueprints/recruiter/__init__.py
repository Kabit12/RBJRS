"""
Recruiter Blueprint
====================
Handles recruiter dashboard, job CRUD, applicant viewing, and candidate ranking.
Will be fully implemented in Milestone 6.
"""

from flask import Blueprint

recruiter_bp = Blueprint('recruiter', __name__, url_prefix='/recruiter')

from app.blueprints.recruiter import routes  # noqa: E402, F401
