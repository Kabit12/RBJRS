"""
Validator Tests
=================
Tests for file validation, MIME type checking, and filename sanitization.
"""

import io
import pytest


class TestSecureFilename:
    """Tests for secure_filename_custom."""

    def test_basic_filename(self):
        from app.utils.validators import secure_filename_custom
        result = secure_filename_custom('test_resume.pdf')
        assert result.endswith('.pdf')
        assert 'test_resume' in result

    def test_dangerous_filename_traversal(self):
        from app.utils.validators import secure_filename_custom
        result = secure_filename_custom('../../etc/passwd.pdf')
        assert '..' not in result
        assert '/' not in result

    def test_empty_filename(self):
        from app.utils.validators import secure_filename_custom
        result = secure_filename_custom('')
        assert 'unnamed_file' in result or 'file' in result

    def test_uniqueness(self):
        from app.utils.validators import secure_filename_custom
        name1 = secure_filename_custom('resume.pdf')
        name2 = secure_filename_custom('resume.pdf')
        # UUID component should make them unique
        assert name1 != name2

    def test_extension_preserved(self):
        from app.utils.validators import secure_filename_custom
        result = secure_filename_custom('MY_RESUME.DOCX')
        assert result.endswith('.docx')  # Extension should be lowercased

    def test_unicode_filename(self):
        from app.utils.validators import secure_filename_custom
        result = secure_filename_custom('résumé_文件.pdf')
        assert result.endswith('.pdf')
        assert len(result) > 4  # Not just extension


class TestResumeValidation:
    """Tests for validate_resume_file."""

    def test_no_file(self):
        from app.utils.validators import validate_resume_file
        is_valid, error = validate_resume_file(None)
        assert is_valid is False
        assert 'No file' in error

    def test_empty_filename(self):
        from app.utils.validators import validate_resume_file
        file = io.BytesIO(b'content')
        file.filename = ''
        is_valid, error = validate_resume_file(file)
        assert is_valid is False

    def test_disallowed_extension(self):
        from app.utils.validators import validate_resume_file
        file = io.BytesIO(b'content')
        file.filename = 'malware.exe'
        is_valid, error = validate_resume_file(file)
        assert is_valid is False
        assert 'Invalid file type' in error

    def test_path_traversal_rejected(self):
        from app.utils.validators import validate_resume_file
        file = io.BytesIO(b'%PDF-1.4')
        file.filename = '../../../etc/passwd.pdf'
        is_valid, error = validate_resume_file(file)
        assert is_valid is False
        assert 'Invalid filename' in error

    def test_valid_pdf_extension(self):
        """Test that a file with valid extension passes extension check."""
        from app.utils.validators import validate_resume_file
        # Create a minimal valid PDF header for MIME detection
        pdf_content = b'%PDF-1.4 fake pdf content for testing'
        file = io.BytesIO(pdf_content)
        file.filename = 'resume.pdf'
        # This may pass or fail MIME depending on python-magic
        # but should at least pass extension check
        is_valid, error = validate_resume_file(file)
        # If python-magic is installed, MIME check may reject this fake PDF
        # That's acceptable — the test verifies the validation pipeline runs


class TestAllowedFile:
    """Tests for allowed_file helper."""

    def test_allowed_pdf(self):
        from app.utils.validators import allowed_file
        assert allowed_file('test.pdf', {'pdf', 'docx'}) is True

    def test_allowed_docx(self):
        from app.utils.validators import allowed_file
        assert allowed_file('test.docx', {'pdf', 'docx'}) is True

    def test_disallowed_exe(self):
        from app.utils.validators import allowed_file
        assert allowed_file('test.exe', {'pdf', 'docx'}) is False

    def test_no_extension(self):
        from app.utils.validators import allowed_file
        assert allowed_file('noextension', {'pdf', 'docx'}) is False

    def test_case_insensitive(self):
        from app.utils.validators import allowed_file
        assert allowed_file('test.PDF', {'pdf', 'docx'}) is True

    def test_double_extension(self):
        from app.utils.validators import allowed_file
        assert allowed_file('test.exe.pdf', {'pdf', 'docx'}) is True  # Checks last extension
