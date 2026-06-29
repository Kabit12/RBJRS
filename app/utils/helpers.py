"""
Helper Utilities
==================
General-purpose helper functions used across the application.
"""

import os
from flask import current_app


def get_upload_path(subfolder='resumes'):
    """
    Returns the absolute path for file uploads.

    Args:
        subfolder: Subfolder within the uploads directory ('resumes', 'logos', 'profiles')

    Returns:
        Absolute path string
    """
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'app/static/uploads')
    path = os.path.join(upload_folder, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def format_datetime(dt, fmt='%B %d, %Y'):
    """Format a datetime object for display."""
    if dt is None:
        return 'N/A'
    return dt.strftime(fmt)


def truncate_text(text, length=150):
    """Truncate text to a specified length with ellipsis."""
    if not text:
        return ''
    if len(text) <= length:
        return text
    return text[:length].rsplit(' ', 1)[0] + '...'
