"""
Test Configuration and Fixtures
=================================
Shared fixtures for all tests. Uses Flask's test client and an
in-memory SQLite database for fast, isolated test execution.

Compatible with Flask-SQLAlchemy 3.x.
"""

import os
import pytest
from app import create_app
from app.extensions import db as _db


@pytest.fixture(scope='session')
def app():
    """Create the Flask application with testing configuration."""
    # Ensure SECRET_KEY is set for testing
    os.environ['SECRET_KEY'] = 'test-secret-key-for-pytest-only'
    os.environ['FLASK_ENV'] = 'testing'

    app = create_app('testing')

    # Override upload folder for tests
    app.config['UPLOAD_FOLDER'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '_test_uploads'
    )
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    yield app

    # Cleanup test uploads
    import shutil
    test_uploads = app.config['UPLOAD_FOLDER']
    if os.path.exists(test_uploads):
        shutil.rmtree(test_uploads, ignore_errors=True)


@pytest.fixture(scope='session')
def _database(app):
    """Create database tables (once per test session)."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.drop_all()


@pytest.fixture(autouse=True)
def db_session(app, _database):
    """
    Per-test database session with automatic cleanup.

    Deletes all rows from every table after each test to ensure
    test isolation. Compatible with Flask-SQLAlchemy 3.x.
    """
    with app.app_context():
        yield _database.session

        # Clean up all data after each test
        _database.session.rollback()
        for table in reversed(_database.metadata.sorted_tables):
            _database.session.execute(table.delete())
        _database.session.commit()


@pytest.fixture
def client(app):
    """Flask test client for making HTTP requests."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Flask CLI test runner."""
    return app.test_cli_runner()


# ── Factory Helpers ──────────────────────────────────────────────────────────

@pytest.fixture
def make_user(app, _database):
    """Factory fixture to create users."""
    from app.extensions import bcrypt
    from app.models.user import User

    def _make_user(email='test@example.com', password='TestPass123!',
                   first_name='Test', last_name='User', role='candidate',
                   is_active=True):
        with app.app_context():
            password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(
                email=email,
                password_hash=password_hash,
                first_name=first_name,
                last_name=last_name,
                role=role,
                is_active=is_active,
            )
            _database.session.add(user)
            _database.session.commit()  # Commit so test client can see it
            return user

    return _make_user


@pytest.fixture
def make_candidate(app, _database, make_user):
    """Factory fixture to create candidate users with profiles."""
    from app.models.candidate import Candidate

    def _make_candidate(email='candidate@example.com', **kwargs):
        user = make_user(email=email, role='candidate', **kwargs)
        candidate = Candidate(user_id=user.id)
        _database.session.add(candidate)
        _database.session.commit()
        return user, candidate

    return _make_candidate


@pytest.fixture
def make_recruiter(app, _database, make_user):
    """Factory fixture to create recruiter users with profiles."""
    from app.models.recruiter import Recruiter

    def _make_recruiter(email='recruiter@example.com', company='TestCorp',
                        is_approved=True, **kwargs):
        user = make_user(email=email, role='recruiter', **kwargs)
        recruiter = Recruiter(
            user_id=user.id,
            company_name=company,
            is_approved=is_approved,
        )
        _database.session.add(recruiter)
        _database.session.commit()
        return user, recruiter

    return _make_recruiter
