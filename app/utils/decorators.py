"""
Authorization Decorators
==========================
Role-based access control decorators for Flask routes.

Design Decision:
- These decorators work in conjunction with Flask-Login's @login_required.
- They check the current_user's role and redirect/abort if unauthorized.
- Using decorators keeps authorization logic DRY across all routes.
- Each role has its own decorator for clarity and explicit access control.

Usage:
    @candidate_bp.route('/dashboard')
    @login_required
    @candidate_required
    def dashboard():
        ...
"""

from functools import wraps
from flask import abort, flash, redirect, url_for
from flask_login import current_user


def role_required(role):
    """
    Generic role-checking decorator factory.

    Args:
        role: Required role string ('candidate', 'recruiter', 'admin')

    Returns:
        Decorator that enforces the role requirement.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('auth.login'))
            if current_user.role != role:
                abort(403)  # Forbidden
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def candidate_required(f):
    """Restricts access to candidates only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if current_user.role != 'candidate':
            flash('Access denied. Candidate account required.', 'danger')
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def recruiter_required(f):
    """Restricts access to recruiters only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if current_user.role != 'recruiter':
            flash('Access denied. Recruiter account required.', 'danger')
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Restricts access to admins only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        if current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            abort(403)
        return f(*args, **kwargs)
    return decorated_function
