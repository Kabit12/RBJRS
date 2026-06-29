"""
Flask Application Factory
===========================
Creates and configures the Flask application instance.

This follows the Application Factory pattern, which:
1. Allows multiple app instances (useful for testing)
2. Defers extension initialization until app creation time
3. Centralizes configuration, extension binding, and blueprint registration
4. Prevents circular imports by not creating the app at module level

The factory performs these steps in order:
1. Create Flask instance
2. Load configuration
3. Initialize extensions (bind to app)
4. Register blueprints (URL routes)
5. Create database tables (if they don't exist)
6. Ensure required directories exist (uploads, ml_models)
"""

import os
from flask import Flask
from app.config import config_map


def create_app(config_name=None):
    """
    Application factory function.

    Args:
        config_name: Configuration environment ('development', 'testing', 'production').
                     Defaults to FLASK_ENV environment variable or 'development'.

    Returns:
        Configured Flask application instance.
    """
    # Create Flask instance
    app = Flask(__name__)

    # Determine configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(config_map.get(config_name, config_map['default']))

    # Initialize extensions
    _init_extensions(app)

    # Register blueprints
    _register_blueprints(app)

    # Setup tasks that need app context
    with app.app_context():
        # Import models so SQLAlchemy discovers them
        from app import models  # noqa: F401

        # Create all database tables
        from app.extensions import db
        db.create_all()

    # Ensure required directories exist
    _ensure_directories(app)

    return app


def _init_extensions(app):
    """Bind all Flask extensions to the application instance."""
    from app.extensions import db, migrate, login_manager, bcrypt, csrf

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    csrf.init_app(app)


def _register_blueprints(app):
    """
    Register all Flask blueprints (route modules).

    Each blueprint handles a specific domain:
    - auth: Registration, login, logout
    - candidate: Candidate dashboard and features
    - recruiter: Recruiter dashboard and features
    - admin: Admin panel
    - api: REST API endpoints (for HTMX and future mobile apps)
    """
    from app.blueprints.auth import auth_bp
    app.register_blueprint(auth_bp)

    from app.blueprints.candidate import candidate_bp
    app.register_blueprint(candidate_bp)

    from app.blueprints.recruiter import recruiter_bp
    app.register_blueprint(recruiter_bp)

    from app.blueprints.admin import admin_bp
    app.register_blueprint(admin_bp)

    from app.blueprints.main import main_bp
    app.register_blueprint(main_bp)


def _ensure_directories(app):
    """Create required directories if they don't exist."""
    directories = [
        app.config.get('UPLOAD_FOLDER', 'app/static/uploads'),
        app.config.get('MODEL_DIR', 'ml_models'),
        os.path.join(app.config.get('UPLOAD_FOLDER', 'app/static/uploads'), 'resumes'),
        os.path.join(app.config.get('UPLOAD_FOLDER', 'app/static/uploads'), 'logos'),
        os.path.join(app.config.get('UPLOAD_FOLDER', 'app/static/uploads'), 'profiles'),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
