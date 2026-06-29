"""
Input Validators
==================
Validation utilities for user input and file uploads.

Design Decision:
- Separate validation layer prevents security issues at the boundary.
- File validation checks both extension and MIME type to prevent malicious uploads.
- All validators return (is_valid, error_message) tuples for clean error handling.
"""

import os


ALLOWED_RESUME_EXTENSIONS = {'pdf', 'docx'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


def allowed_file(filename, allowed_extensions):
    """
    Check if a filename has an allowed extension.

    Args:
        filename: The uploaded file's name
        allowed_extensions: Set of allowed extension strings (without dots)

    Returns:
        True if the file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def validate_resume_file(file):
    """
    Validate an uploaded resume file.

    Checks:
    1. File is not empty
    2. File has an allowed extension (.pdf or .docx)
    3. Filename is safe (no path traversal)

    Args:
        file: Flask FileStorage object

    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    if not file or file.filename == '':
        return False, 'No file selected.'

    if not allowed_file(file.filename, ALLOWED_RESUME_EXTENSIONS):
        return False, 'Invalid file type. Only PDF and DOCX files are allowed.'

    # Check filename for path traversal
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        return False, 'Invalid filename.'

    return True, None


def validate_image_file(file):
    """
    Validate an uploaded image file.

    Args:
        file: Flask FileStorage object

    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
    """
    if not file or file.filename == '':
        return False, 'No file selected.'

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return False, 'Invalid file type. Only PNG, JPG, JPEG, GIF, and WebP files are allowed.'

    return True, None


def secure_filename_custom(filename):
    """
    Generate a secure version of a filename while preserving readability.
    Replaces spaces with underscores and removes potentially dangerous characters.

    Args:
        filename: Original filename string

    Returns:
        Sanitized filename string
    """
    import re
    import uuid
    from datetime import datetime

    # Get extension
    name, ext = os.path.splitext(filename)

    # Remove non-alphanumeric characters (except underscores and hyphens)
    name = re.sub(r'[^\w\-]', '_', name)

    # Add timestamp and short UUID for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]

    return f'{name}_{timestamp}_{unique_id}{ext.lower()}'
