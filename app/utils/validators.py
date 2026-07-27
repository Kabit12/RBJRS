"""
Input Validators
==================
Validation utilities for user input and file uploads.

Design Decision:
- Defense-in-depth: validates BOTH file extension AND MIME type (content sniffing).
- Extension-only checks are trivially bypassed by renaming files.
- python-magic reads the file's magic bytes to determine true content type.
- werkzeug.utils.secure_filename is used as the base sanitizer (battle-tested),
  then augmented with timestamp + UUID for uniqueness.
- All validators return (is_valid, error_message) tuples for clean error handling.
"""

import os
import re
import uuid
from datetime import datetime

from werkzeug.utils import secure_filename as werkzeug_secure_filename

# Try to import python-magic for MIME validation (defense-in-depth)
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


ALLOWED_RESUME_EXTENSIONS = {'pdf', 'docx'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# MIME types corresponding to allowed resume extensions
ALLOWED_RESUME_MIMES = {
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}

# MIME types corresponding to allowed image extensions
ALLOWED_IMAGE_MIMES = {
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
}


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


def _validate_mime_type(file, allowed_mimes):
    """
    Validate file content by reading magic bytes (MIME sniffing).

    Reads the first 2048 bytes to detect the true file type,
    regardless of what the extension claims. First uses native magic byte
    header signatures, falling back to python-magic if available.

    Args:
        file: Flask FileStorage object (seek position is restored after read)
        allowed_mimes: Set of allowed MIME type strings

    Returns:
        Tuple of (is_valid: bool, detected_mime: str)
    """
    try:
        file_bytes = file.read(2048)
        file.seek(0)  # Reset file pointer for subsequent reads
    except Exception:
        file.seek(0)
        return False, 'error reading file bytes'

    # Security Fix: Native magic byte signature checks for robust MIME detection
    if file_bytes.startswith(b'%PDF-'):
        detected_mime = 'application/pdf'
    elif file_bytes.startswith(b'PK\x03\x04'):
        # OpenXML format (DOCX) or Zip container
        detected_mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        detected_mime = 'image/png'
    elif file_bytes.startswith(b'\xff\xd8\xff'):
        detected_mime = 'image/jpeg'
    elif file_bytes.startswith(b'GIF87a') or file_bytes.startswith(b'GIF89a'):
        detected_mime = 'image/gif'
    elif file_bytes.startswith(b'RIFF') and len(file_bytes) >= 12 and file_bytes[8:12] == b'WEBP':
        detected_mime = 'image/webp'
    elif HAS_MAGIC:
        try:
            detected_mime = magic.from_buffer(file_bytes, mime=True)
        except Exception:
            return False, 'error detecting MIME type'
    else:
        return True, 'unknown (python-magic not installed)'

    return detected_mime in allowed_mimes, detected_mime


def validate_resume_file(file):
    """
    Validate an uploaded resume file with defense-in-depth checks.

    Checks (in order):
    1. File is not empty
    2. File has an allowed extension (.pdf or .docx)
    3. Filename has no path traversal characters
    4. MIME type of actual content matches the claimed extension

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

    # MIME type validation (defense-in-depth)
    mime_valid, detected_mime = _validate_mime_type(file, ALLOWED_RESUME_MIMES)
    if not mime_valid:
        return False, (
            f'File content does not match the expected format. '
            f'Detected: {detected_mime}. Only PDF and DOCX content is allowed.'
        )

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

    # MIME type validation
    mime_valid, detected_mime = _validate_mime_type(file, ALLOWED_IMAGE_MIMES)
    if not mime_valid:
        return False, (
            f'File content does not match the expected image format. '
            f'Detected: {detected_mime}.'
        )

    return True, None


def secure_filename_custom(filename):
    """
    Generate a secure, unique filename using werkzeug's battle-tested sanitizer.

    Process:
    1. Apply werkzeug.utils.secure_filename to strip dangerous characters
    2. Append timestamp + short UUID for uniqueness
    3. Preserve the original (lowercased) extension

    Args:
        filename: Original filename string

    Returns:
        Sanitized, unique filename string
    """
    # Use werkzeug's secure_filename as the base (handles path traversal, unicode, etc.)
    safe_name = werkzeug_secure_filename(filename)

    if not safe_name:
        safe_name = 'unnamed_file'

    # Split into name and extension
    name, ext = os.path.splitext(safe_name)

    # Clean the name part (extra safety)
    name = re.sub(r'[^\w\-]', '_', name)

    if not name:
        name = 'file'

    # Add timestamp and short UUID for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]

    return f'{name}_{timestamp}_{unique_id}{ext.lower()}'
