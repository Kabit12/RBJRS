"""Main public routes — landing page."""

from flask import render_template
from app.blueprints.main import main_bp


@main_bp.route('/')
def index():
    """Landing page — the first thing users see."""
    return render_template('landing.html')
