"""
Candidate Blueprint
====================
Handles candidate dashboard, resume upload, recommendations, and applications.
Will be fully implemented in Milestone 6.
"""

from flask import Blueprint

candidate_bp = Blueprint('candidate', __name__, url_prefix='/candidate')

from app.blueprints.candidate import routes  # noqa: E402, F401
from app.blueprints.candidate import file_routes  # noqa: E402, F401
