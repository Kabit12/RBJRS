"""
Authentication Service
========================
Business logic for user authentication operations.

Design Decision:
- Service layer separates business logic from route handlers (MVC pattern).
- All database operations and password hashing happen here, not in routes.
- Returns (success, result_or_error) tuples for clean error handling.
- The route handler only deals with HTTP concerns (request/response/redirect).
"""

from app.extensions import db, bcrypt
from app.models.user import User
from app.models.candidate import Candidate
from app.models.recruiter import Recruiter


def register_candidate(form_data):
    """
    Register a new candidate user.

    Process:
    1. Hash the password with bcrypt
    2. Create User record with role='candidate'
    3. Create associated Candidate profile
    4. Commit both to database

    Args:
        form_data: Dictionary with registration form fields

    Returns:
        Tuple of (success: bool, user_or_error: User | str)
    """
    try:
        # Hash password with bcrypt (automatically handles salt generation)
        password_hash = bcrypt.generate_password_hash(
            form_data['password']
        ).decode('utf-8')

        # Create user
        user = User(
            email=form_data['email'].lower().strip(),
            password_hash=password_hash,
            first_name=form_data['first_name'].strip(),
            last_name=form_data['last_name'].strip(),
            role='candidate'
        )
        db.session.add(user)
        db.session.flush()  # Get user.id before creating profile

        # Create candidate profile
        candidate = Candidate(
            user_id=user.id,
            phone=form_data.get('phone', '').strip() or None,
            location=form_data.get('location', '').strip() or None
        )
        db.session.add(candidate)
        db.session.commit()

        return True, user

    except Exception as e:
        db.session.rollback()
        return False, f'Registration failed: {str(e)}'


def register_recruiter(form_data):
    """
    Register a new recruiter user.

    Process:
    1. Hash the password with bcrypt
    2. Create User record with role='recruiter'
    3. Create associated Recruiter profile with company information
    4. Commit both to database

    Args:
        form_data: Dictionary with registration form fields

    Returns:
        Tuple of (success: bool, user_or_error: User | str)
    """
    try:
        password_hash = bcrypt.generate_password_hash(
            form_data['password']
        ).decode('utf-8')

        user = User(
            email=form_data['email'].lower().strip(),
            password_hash=password_hash,
            first_name=form_data['first_name'].strip(),
            last_name=form_data['last_name'].strip(),
            role='recruiter'
        )
        db.session.add(user)
        db.session.flush()

        recruiter = Recruiter(
            user_id=user.id,
            company_name=form_data['company_name'].strip(),
            industry=form_data.get('industry', '').strip() or None,
            company_size=form_data.get('company_size', '').strip() or None,
            phone=form_data.get('phone', '').strip() or None,
            location=form_data.get('location', '').strip() or None
        )
        db.session.add(recruiter)
        db.session.commit()

        return True, user

    except Exception as e:
        db.session.rollback()
        return False, f'Registration failed: {str(e)}'


def authenticate_user(email, password):
    """
    Authenticate a user with email and password.

    Process:
    1. Look up user by email
    2. Verify password against stored bcrypt hash
    3. Check if account is active

    Args:
        email: User's email address
        password: Plain-text password to verify

    Returns:
        Tuple of (success: bool, user_or_error: User | str)
    """
    user = User.query.filter_by(email=email.lower().strip()).first()

    if not user:
        return False, 'Invalid email or password.'

    if not bcrypt.check_password_hash(user.password_hash, password):
        return False, 'Invalid email or password.'

    if not user.is_active:
        return False, 'Your account has been deactivated. Please contact support.'

    return True, user


def create_admin_user(email, password, first_name='Admin', last_name='User'):
    """
    Create an admin user (for initial setup/seeding).

    Args:
        email: Admin email
        password: Admin password
        first_name: Admin first name
        last_name: Admin last name

    Returns:
        Tuple of (success: bool, user_or_error: User | str)
    """
    try:
        existing = User.query.filter_by(email=email.lower().strip()).first()
        if existing:
            return False, 'Admin user already exists.'

        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

        user = User(
            email=email.lower().strip(),
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            role='admin'
        )
        db.session.add(user)
        db.session.commit()

        return True, user

    except Exception as e:
        db.session.rollback()
        return False, f'Failed to create admin: {str(e)}'
