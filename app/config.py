"""
Application Configuration Module
=================================
Defines configuration classes for different environments (Development, Testing, Production).
Uses python-dotenv to load environment variables from .env file.

Design Decision:
- Class-based config with inheritance allows clean environment switching
- All sensitive values come from environment variables (never hardcoded)
- SQLite default for development means zero database setup required
- MySQL can be enabled by simply changing DATABASE_URL in .env
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config:
    """Base configuration shared across all environments."""

    # Flask Core — SECRET_KEY is REQUIRED. No fallback.
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError(
            'FATAL: SECRET_KEY environment variable is not set. '
            'Set it in your .env file: SECRET_KEY=<your-random-secret-at-least-32-chars>'
        )

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "instance", "rbjrs.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set True to see SQL queries in console

    # File Upload — OUTSIDE web root for security (no static serving of PII)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}

    # ML Models Directory
    MODEL_DIR = os.path.join(BASE_DIR, os.environ.get('MODEL_DIR', 'ml_models'))

    # Session Configuration
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

    # WTForms CSRF Protection
    WTF_CSRF_ENABLED = True

    # Google OAuth (for "Sign in with Google" on candidate side)
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Show SQL queries during development


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # In-memory DB for tests
    WTF_CSRF_ENABLED = False  # Disable CSRF for testing
    RATELIMIT_ENABLED = False  # Disable rate limiting in tests


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    # In production, DATABASE_URL should point to MySQL
    SESSION_COOKIE_SECURE = True      # Cookies only sent over HTTPS
    SESSION_COOKIE_HTTPONLY = True     # Cookies inaccessible to JavaScript
    SESSION_COOKIE_SAMESITE = 'Lax'   # CSRF protection for cookies


# Configuration mapping for easy access
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
