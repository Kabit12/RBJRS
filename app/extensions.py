"""
Flask Extensions Module
========================
Centralizes initialization of all Flask extensions.

Design Decision:
- Extensions are instantiated here WITHOUT the app object (lazy initialization).
- They are bound to the app later in the app factory (__init__.py) via init_app().
- This avoids circular imports and follows the Flask application factory pattern.
- Each extension serves a specific architectural purpose documented below.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect

# Database ORM - Maps Python classes to database tables
# Provides query interface, relationship management, and transaction handling
db = SQLAlchemy()

# Database Migrations - Tracks schema changes via Alembic
# Allows safe, versioned database schema evolution
migrate = Migrate()

# Authentication Manager - Handles user session management
# Provides login_required decorator, current_user proxy, remember-me functionality
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Redirect endpoint for unauthenticated users
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Password Hashing - Bcrypt for secure password storage
# Bcrypt is intentionally slow (configurable work factor) to resist brute-force attacks
bcrypt = Bcrypt()

# CSRF Protection - Prevents Cross-Site Request Forgery attacks
# Automatically validates CSRF tokens on form submissions
csrf = CSRFProtect()
